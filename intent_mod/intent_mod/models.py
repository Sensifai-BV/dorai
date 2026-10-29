"""Locate the intent GGUF model.

Search roots, in order (the first root holding the file wins):
the ``models_dir`` parameter, $DORAI_MODELS_DIR, the package's own ``models/`` dir
(intent_mod/models in the source tree, share/intent_mod/models when installed),
~/dorai/models, /root/.cache/dorai/models. Inside a root the file is looked up as
``<root>/<name>`` and then ``<root>/<subdir>/<name>`` (one level, sorted).
``name`` may also be an absolute path, or a huggingface.co file URL.

When the file is in no root, ``resolve_model`` downloads it from the Hugging Face repo
(``hf_repo``, default sensifai/dorai-intent-model) into ``models_dir`` or ~/dorai/models,
resuming a partial ``.part`` file and checking the sha256 the Hub reports. A private repo
needs $HF_TOKEN.
"""
from __future__ import annotations

import glob
import hashlib
import logging
import os
import re
import urllib.error
import urllib.request

DEFAULT_LORA = ""   # optional runtime adapter (slower on ARM); the default model already has it merged
DEFAULT_MODEL = "intent-qwen35-4b-Q4_0.gguf"   # Qwen3.5-4B + dorai Artaban LoRA, merged, Q4_0 (train/)
HF_REPO = "sensifai/dorai-intent-model"          # where DEFAULT_MODEL is published
_HF_URL = re.compile(r"^https?://huggingface\.co/([^/]+/[^/]+)/(?:blob|resolve)/([^/]+)/(.+?)(?:\?.*)?$")


def _package_models_dirs():
    here = os.path.dirname(os.path.abspath(__file__))
    out = [os.path.normpath(os.path.join(here, "..", "models"))]   # source tree / symlink-install
    try:
        from ament_index_python.packages import get_package_share_directory
        out.append(os.path.join(get_package_share_directory("intent_mod"), "models"))
    except Exception:  # not a ROS environment, or package not installed
        pass
    return out


def default_roots(extra=""):
    roots = [extra, os.environ.get("DORAI_MODELS_DIR", ""), *_package_models_dirs(),
             os.path.join(os.path.expanduser("~"), "dorai", "models"),
             "/root/.cache/dorai/models"]
    seen, out = set(), []
    for r in roots:
        if r and os.path.realpath(os.path.expanduser(r)) not in seen:
            seen.add(os.path.realpath(os.path.expanduser(r)))
            out.append(r)
    return out


def find_model(name=DEFAULT_MODEL, models_dir="", roots=None):
    """Path of the GGUF, or None when it is not installed in any root."""
    name = name or DEFAULT_MODEL
    if os.path.isabs(name):
        return name if os.path.isfile(name) else None
    for root in (roots if roots is not None else default_roots(models_dir)):
        root = os.path.expanduser(root)
        p = os.path.join(root, name)
        if os.path.isfile(p):
            return p
        hits = sorted(glob.glob(os.path.join(glob.escape(root), "*", name)))
        if hits:
            return hits[0]
    return None


def download_dir(models_dir=""):
    return os.path.expanduser(models_dir or os.path.join("~", "dorai", "models"))


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def _hf_sha256(url, hdr):
    """sha256 the Hub reports for an LFS/xet file (X-Linked-ETag on the pre-redirect response), or ''."""
    req = urllib.request.Request(url, headers=hdr, method="HEAD")
    try:
        r = urllib.request.build_opener(_NoRedirect).open(req, timeout=30)
    except urllib.error.HTTPError as e:              # the 302 itself arrives as an "error"
        r = e
    if r.code in (401, 403, 404):
        raise IOError(f"HTTP {r.code} for {url}" + (" (private repo? set HF_TOKEN)" if r.code != 404 else ""))
    return (r.headers.get("X-Linked-ETag") or "").strip('"')


def hf_download(repo, filename, dest_dir, revision="main", log=None, chunk=1 << 22):
    """Download <repo>/<filename>@revision to dest_dir/<basename>; resume, verify sha256. -> path."""
    log = log or logging.getLogger("intent_mod")
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, os.path.basename(filename))
    part = dest + ".part"
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
    hdr = {"User-Agent": "dorai-intent_mod"}
    if os.environ.get("HF_TOKEN"):
        hdr["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    want = _hf_sha256(url, hdr)
    have = os.path.getsize(part) if os.path.exists(part) else 0
    req = urllib.request.Request(url, headers={**hdr, **({"Range": f"bytes={have}-"} if have else {})})
    with urllib.request.urlopen(req, timeout=60) as r:
        if have and r.status != 206:                    # server ignored Range: start over
            have = 0
        total = have + int(r.headers.get("Content-Length") or 0)
        log.info(f"downloading {url} -> {dest} ({total / 1e9:.2f} GB{f', resuming at {have / 1e9:.2f} GB' if have else ''})")
        h = hashlib.sha256()
        if have:
            with open(part, "rb") as f:
                for b in iter(lambda: f.read(chunk), b""):
                    h.update(b)
        done, next_log = have, have + total // 10
        with open(part, "ab" if have else "wb") as f:
            for b in iter(lambda: r.read(chunk), b""):
                f.write(b)
                h.update(b)
                done += len(b)
                if done >= next_log:
                    log.info(f"  {100 * done / max(1, total):.0f} % of {os.path.basename(dest)}")
                    next_log += total // 10
    got = h.hexdigest()
    if re.fullmatch(r"[0-9a-f]{64}", want or "") and got != want:
        os.remove(part)
        raise IOError(f"sha256 mismatch for {url}: got {got}, Hub says {want}")
    os.replace(part, dest)
    log.info(f"downloaded {dest} (sha256 {got[:12]}…)")
    return dest


def resolve_model(name=DEFAULT_MODEL, models_dir="", hf_repo=HF_REPO, download=True, log=None):
    """Path of the GGUF: an absolute path, a file in a models root, or - when it is in none and
    `download` is on - fetched from Hugging Face (`name` as a huggingface.co URL, else
    `hf_repo`/`name`). None when it cannot be found or fetched."""
    log = log or logging.getLogger("intent_mod")
    name = (name or DEFAULT_MODEL).strip()
    m = _HF_URL.match(name)
    repo, rev, fname = (m.group(1), m.group(2), m.group(3)) if m else (hf_repo, "main", name)
    if not m:
        path = find_model(os.path.expanduser(name), models_dir)
        if path or os.path.isabs(os.path.expanduser(name)):
            return path                                  # explicit paths are never downloaded
    else:
        path = find_model(os.path.basename(fname), models_dir)
        if path:
            return path
    if not (download and repo):
        return None
    try:
        return hf_download(repo, fname, download_dir(models_dir), rev, log=log)
    except Exception as e:  # noqa: BLE001  (offline, 401/404, disk full ...)
        log.warning(f"could not download {fname} from huggingface.co/{repo}: {e!r}")
        return None

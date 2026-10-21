import os
from glob import glob

from setuptools import setup

package_name = "stt_mod"


def _model_files():
    """Install models/<dir>/* (and test_wavs/) to share/stt_mod/models/<dir>/.
    With colcon --symlink-install these are symlinks, so nothing is copied."""
    out = []
    for d in sorted(glob("models/*/")):
        for sub in ("", "test_wavs/"):
            files = [f for f in glob(d + sub + "*") if os.path.isfile(f)]
            if files:
                out.append(("share/" + package_name + "/" + d + sub, files))
    return out

setup(
    name=package_name,
    version="0.2.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ] + _model_files(),
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Sensifai",
    maintainer_email="info@sensifai.com",
    description="dorai Stage 2: streaming speech-to-text with NVIDIA Nemotron "
                "0.6B (NeMo, sherpa-onnx) + dorai fine-tuned adapters.",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "stt = stt_mod.node:main",
            "record_clean = stt_mod.record_clean:main",
            "bake_adapter = stt_mod.bake_adapter:main",
            "transcribe = stt_mod.transcribe:main",
            "evaluate = stt_mod.evaluate:main",
        ],
    },
)

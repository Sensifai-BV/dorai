import os
from glob import glob

from setuptools import setup

package_name = "intent_mod"


def _model_files():
    """Install models/*.gguf and models/<dir>/*.gguf to share/intent_mod/models/[<dir>/].
    With colcon --symlink-install these are symlinks, so nothing is copied."""
    out = []
    flat = [f for f in glob("models/*.gguf") if os.path.isfile(f)]
    if flat:
        out.append(("share/" + package_name + "/models", flat))
    for d in sorted(glob("models/*/")):
        files = [f for f in glob(d + "*.gguf") if os.path.isfile(f)]
        if files:
            out.append(("share/" + package_name + "/" + d, files))
    return out


setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md", "intent_design.md"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ] + _model_files(),
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Sensifai",
    maintainer_email="info@sensifai.com",
    description="dorai Stage 3: stop/wake gate + grammar-constrained LLM intent parser "
                "(llama.cpp) with a rule-based fallback.",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "intent = intent_mod.node:main",
        ],
    },
)

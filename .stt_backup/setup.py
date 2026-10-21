from setuptools import setup

package_name = "stt_mod"

setup(
    name=package_name,
    version="0.1.0",
    py_modules=["stt"],
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Sensifai",
    maintainer_email="info@sensifai.com",
    description="dorai Stage 2: Speech-to-Text (Nemotron streaming ASR via sherpa-onnx, Whisper, Vosk).",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "stt = stt:main",
        ],
    },
)

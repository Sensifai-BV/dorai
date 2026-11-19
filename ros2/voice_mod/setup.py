from setuptools import setup

package_name = "voice_mod"

setup(
    name=package_name,
    version="0.1.0",
    # Flat layout: voice.py is a top-level module. console_scripts below
    # points `ros2 run voice_mod voice` at voice:main.
    py_modules=["voice", "record_debug"],
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name, ["dorai_beamformer.ort"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Sensifai",
    maintainer_email="info@sensifai.com",
    description=(
        "dorai real-time mic-array front-end: capture + drift-corrected 16 kHz "
        "resampling + dorai_beamformer.ort beamforming, published in N-second frames."
    ),
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "voice = voice:main",
            "record = record_debug:main",
        ],
    },
)

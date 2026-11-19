from setuptools import setup

package_name = "doa_mod"

setup(
    name=package_name,
    version="0.1.0",
    # Flat layout, matching voice_mod/stt_mod: doa.py is a top-level module and
    # the console_script below points `ros2 run doa_mod doa` at doa:main.
    #
    # doa_core and vad are VENDORED COPIES of doa/doa_core.py and doa/vad.py,
    # byte-identical by design. A ROS package cannot reach outside its own
    # directory at build time, so the validated estimator has to live here too.
    # test/test_doa_mod.py::test_vendored_core_matches_research_tree checksums
    # them against doa/ and fails if the two ever fork.
    py_modules=["doa", "doa_core", "vad"],
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name, ["vad.onnx"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Sensifai",
    maintainer_email="info@sensifai.com",
    description=(
        "dorai direction-of-arrival: mic-array capture + VAD gate + pooled "
        "SRP-PHAT bearing, published in degrees on /dorai_doa."
    ),
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "doa = doa:main",
        ],
    },
)

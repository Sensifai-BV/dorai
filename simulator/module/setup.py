from setuptools import setup

package_name = "sim"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Sensifai",
    maintainer_email="info@sensifai.com",
    description=(
        "Simulated microphone arrays for dorai: renders rooms with pyroomacoustics and plays "
        "them into the ROS graph as virtual microphones, with the truth to score against."
    ),
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "virtual_mic = sim.virtual_mic:main",
            "render = sim.render:main",
            "score_doa = sim.score_doa:main",
        ],
    },
)

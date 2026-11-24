from setuptools import setup

package_name = "dorai_simulator"

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
        "DorAI virtual acoustic lab, ROS 2 side: plays rendered scenes as the robot's microphones "
        "and publishes the truth to score against."
    ),
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "virtual_mic = dorai_simulator.simulator_node:main",
        ],
    },
)

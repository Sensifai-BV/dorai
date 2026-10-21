"""ros2 launch intent_mod intent.launch.py [engine:=auto|llm|rules] [model:=<gguf>] [models_dir:=] [require_wake:=true] [num_threads:=2] [locations_file:=]"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory("intent_mod"), "config", "intent.yaml")
    args = [("engine", "auto"), ("model", "intent-lfm25-1.2b-ft-Q4_K_M.gguf"), ("models_dir", ""),
            ("num_threads", "2"), ("require_wake", "true"), ("locations_file", ""), ("debug", "false")]
    return LaunchDescription(
        [DeclareLaunchArgument(k, default_value=v) for k, v in args] + [
            Node(package="intent_mod", executable="intent", name="intent_mod", output="screen",
                 parameters=[cfg] + [{k: LaunchConfiguration(k)} for k, _ in args]),
        ])

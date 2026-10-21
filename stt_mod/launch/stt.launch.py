"""ros2 launch stt_mod stt.launch.py [latency_ms:=1120] [adapter:=mc-ref] [num_threads:=2] [metrics:=true] [metrics_file:=/tmp/stt_metrics.jsonl]"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory("stt_mod"), "config", "stt.yaml")
    args = [("latency_ms", "1120"), ("adapter", "mc-ref"), ("num_threads", "2"),
            ("models_dir", ""), ("metrics", "false"), ("metrics_file", ""), ("debug", "false")]
    return LaunchDescription(
        [DeclareLaunchArgument(k, default_value=v) for k, v in args] + [
            Node(package="stt_mod", executable="stt", name="stt_mod", output="screen",
                 parameters=[cfg] + [{k: LaunchConfiguration(k)} for k, _ in args]),
        ])

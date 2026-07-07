#!/bin/bash
# Source the ROS base install and the built dorai workspace overlay, then run
# whatever command the container was given (compose `command:` or the CMD).
set -e
source /opt/ros/jazzy/setup.bash
source /ros2_ws/install/setup.bash
exec "$@"

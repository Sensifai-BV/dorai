"""rosmock — a minimal in-process stand-in for rclpy, so doa_mod can be tested
without a ROS 2 installation.

The node's real defects live in parameter handling, geometry resolution, the
publish gate and shutdown ordering -- all of which are plain Python that a
mocked rclpy exercises perfectly well. Requiring a full ROS install to reach
them would mean they simply never got tested on a laptop.

install() must be called BEFORE `import doa`. It registers fake modules in
sys.modules; doa.py then imports them believing they are the real thing.
Captured messages are readable from each publisher's `.sent` list.
"""
from __future__ import annotations

import sys
import types


class Publisher:
    def __init__(self, msg_type, topic, qos):
        self.msg_type = msg_type
        self.topic = topic
        self.qos = qos
        self.sent = []

    def publish(self, msg):
        self.sent.append(msg)


class Subscription:
    def __init__(self, msg_type, topic, cb, qos):
        self.msg_type = msg_type
        self.topic = topic
        self.callback = cb
        self.qos = qos


class Logger:
    def __init__(self):
        self.records = []

    def _add(self, level, msg):
        self.records.append((level, str(msg)))

    def info(self, m):
        self._add('info', m)

    def warning(self, m):
        self._add('warn', m)

    def warn(self, m):
        self._add('warn', m)

    def error(self, m):
        self._add('error', m)

    def debug(self, m):
        self._add('debug', m)

    def text(self):
        return "\n".join(f"[{lv}] {m}" for lv, m in self.records)


class _Param:
    def __init__(self, value):
        self.value = value


class Node:
    """Enough of rclpy.node.Node for doa_mod: params, pubs, subs, logger."""

    def __init__(self, name):
        self._name = name
        self._params = {}
        self._logger = Logger()
        self.publishers_by_topic = {}
        self.subscriptions_by_topic = {}
        # Overrides injected by the test before construction.
        self._overrides = dict(getattr(type(self), '_pending_overrides', {}))

    def declare_parameter(self, name, value=None, descriptor=None):
        self._params[name] = self._overrides.get(name, value)
        return _Param(self._params[name])

    def get_parameter(self, name):
        return _Param(self._params[name])

    def get_logger(self):
        return self._logger

    def create_publisher(self, msg_type, topic, qos=10):
        p = Publisher(msg_type, topic, qos)
        self.publishers_by_topic[topic] = p
        return p

    def create_subscription(self, msg_type, topic, cb, qos=10):
        s = Subscription(msg_type, topic, cb, qos)
        self.subscriptions_by_topic[topic] = s
        return s

    def destroy_node(self):
        pass


def _msg_class(name, fields):
    def __init__(self, **kw):
        for f in fields:
            setattr(self, f, kw.get(f, None))
        for k, v in kw.items():
            setattr(self, k, v)
    return type(name, (), {"__init__": __init__})


def install():
    """Register the fake ROS modules. Call before importing doa."""
    rclpy = types.ModuleType("rclpy")
    rclpy.init = lambda *a, **k: None
    rclpy.shutdown = lambda *a, **k: None
    rclpy.spin = lambda node: None
    rclpy_node = types.ModuleType("rclpy.node")
    rclpy_node.Node = Node
    rclpy.node = rclpy_node

    std_msgs = types.ModuleType("std_msgs")
    std_msgs_msg = types.ModuleType("std_msgs.msg")
    std_msgs_msg.Float32 = _msg_class("Float32", ["data"])
    std_msgs_msg.String = _msg_class("String", ["data"])
    std_msgs_msg.MultiArrayDimension = _msg_class(
        "MultiArrayDimension", ["label", "size", "stride"])

    class _Layout:
        def __init__(self):
            self.dim = []
            self.data_offset = 0

    class Float32MultiArray:
        def __init__(self, **kw):
            self.data = kw.get("data", [])
            self.layout = _Layout()

    std_msgs_msg.Float32MultiArray = Float32MultiArray
    std_msgs.msg = std_msgs_msg

    rcl_interfaces = types.ModuleType("rcl_interfaces")
    rcl_msg = types.ModuleType("rcl_interfaces.msg")
    rcl_msg.ParameterDescriptor = _msg_class(
        "ParameterDescriptor", ["dynamic_typing"])
    rcl_interfaces.msg = rcl_msg

    for name, mod in (("rclpy", rclpy), ("rclpy.node", rclpy_node),
                      ("std_msgs", std_msgs), ("std_msgs.msg", std_msgs_msg),
                      ("rcl_interfaces", rcl_interfaces),
                      ("rcl_interfaces.msg", rcl_msg)):
        sys.modules[name] = mod
    return rclpy


def set_overrides(params):
    """Parameter overrides applied by the next Node construction."""
    Node._pending_overrides = dict(params)

"""Minimal fake rclpy / std_msgs / rcl_interfaces so intent_mod.node imports and
runs without a ROS install (offline tests only)."""
import sys
import types


class _Logger:
    def __init__(self):
        self.lines = []

    def _log(self, lvl, msg, **kw):
        self.lines.append((lvl, msg))

    def info(self, m, **kw): self._log("info", m)
    def warning(self, m, **kw): self._log("warning", m)
    def error(self, m, **kw): self._log("error", m)


class _Param:
    def __init__(self, v): self.value = v


class _Pub:
    def __init__(self, topic): self.topic, self.msgs = topic, []
    def publish(self, m):
        self.msgs.append(m.data)


class Node:
    overrides = {}

    def __init__(self, name):
        self._params, self._log, self.pubs, self.subs, self.timers = {}, _Logger(), {}, {}, []

    def declare_parameter(self, k, v, desc=None):
        self._params[k] = Node.overrides.get(k, v)

    def get_parameter(self, k): return _Param(self._params[k])
    def get_logger(self): return self._log

    def create_publisher(self, typ, topic, depth):
        self.pubs[topic] = _Pub(topic)
        return self.pubs[topic]

    def create_subscription(self, typ, topic, cb, depth):
        self.subs[topic] = cb

    def create_timer(self, period, cb): self.timers.append(cb)
    def destroy_node(self): pass


class String:
    def __init__(self, data=""): self.data = data


class _Layout:
    data_offset = 0


class Float32:
    def __init__(self, data=0.0): self.data = float(data)


class Float32MultiArray:
    def __init__(self, data=None, data_offset=4):
        self.data = list(data or [])
        self.layout = _Layout()
        self.layout.data_offset = data_offset


class ParameterDescriptor:
    def __init__(self, **kw): pass


def install():
    m = {n: types.ModuleType(n) for n in (
        "rclpy", "rclpy.node", "std_msgs", "std_msgs.msg", "rcl_interfaces", "rcl_interfaces.msg")}
    m["rclpy.node"].Node = Node
    m["rclpy"].node = m["rclpy.node"]
    m["std_msgs.msg"].String = String
    m["std_msgs.msg"].Float32 = Float32
    m["std_msgs.msg"].Float32MultiArray = Float32MultiArray
    m["rcl_interfaces.msg"].ParameterDescriptor = ParameterDescriptor
    sys.modules.update(m)

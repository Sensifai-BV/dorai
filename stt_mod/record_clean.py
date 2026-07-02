#!/usr/bin/env python3
"""
Utility script to record audio from /dorai_clean_audio and save to clean_output.wav
for debugging sound quality. Uses built-in wave library (no external dependencies).
"""

import sys
import wave
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class AudioRecorder(Node):
    def __init__(self):
        super().__init__("audio_recorder")
        
        # Declare parameters for easy configuration
        self.declare_parameter("duration", 20.0)
        self.declare_parameter("filename", "clean_output.wav")
        
        duration = self.get_parameter("duration").get_parameter_value().double_value
        filename = self.get_parameter("filename").get_parameter_value().string_value

        self.filename = filename
        self.max_samples = int(16000 * duration)
        self.samples_collected = 0
        self.audio_data = []

        self.sub = self.create_subscription(
            Float32MultiArray,
            "/dorai_clean_audio",
            self.callback,
            10
        )
        self.get_logger().info(f"Recording from /dorai_clean_audio to '{filename}' for {duration} seconds...")

    def callback(self, msg):
        data_offset = msg.layout.data_offset
        if len(msg.data) < data_offset:
            return
        
        chunk = np.array(msg.data[data_offset:], dtype=np.float32)
        self.audio_data.append(chunk)
        self.samples_collected += len(chunk)
        
        self.get_logger().info(f"Recorded {self.samples_collected}/{self.max_samples} samples...", throttle_duration_sec=1.0)
        
        if self.samples_collected >= self.max_samples:
            self.save_and_exit()

    def save_and_exit(self):
        full_audio = np.concatenate(self.audio_data)[:self.max_samples]
        
        # Convert float32 [-1.0, 1.0] to int16
        full_audio = np.clip(full_audio, -1.0, 1.0)
        pcm16 = (full_audio * 32767.0).astype(np.int16)
        
        # Save using wave
        with wave.open(self.filename, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)  # 16-bit
            w.setframerate(16000)
            w.writeframes(pcm16.tobytes())
            
        self.get_logger().info(f"Successfully saved {self.filename}! Exiting.")
        sys.exit(0)


def main():
    rclpy.init()
    node = AudioRecorder()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

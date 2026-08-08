#!/usr/bin/env python3
import sys
import os
import time
import argparse
import logging
from datetime import datetime
import types
import wave
import numpy as np
import threading
import sounddevice as sd

# --- Mock ROS 2 ---
# We mock the rclpy and std_msgs modules so that we can import the pipeline
# components from voice_mod.voice on a Mac that doesn't have ROS 2 installed.
sys.modules['rclpy'] = types.ModuleType('rclpy')
sys.modules['rclpy.node'] = types.ModuleType('rclpy.node')
sys.modules['rclpy.node'].Node = object
sys.modules['std_msgs'] = types.ModuleType('std_msgs')
sys.modules['std_msgs.msg'] = types.ModuleType('std_msgs.msg')
sys.modules['std_msgs.msg'].Float32MultiArray = object
sys.modules['std_msgs.msg'].MultiArrayDimension = object
sys.modules['std_msgs.msg'].String = object

# Now we can safely import the pipeline logic from the ROS node script
try:
    from voice_mod.voice import (
        detect_microphones, MicChannel, Beamformer, ContinuousEnhancer,
        OUTPUT_RATE, BLOCK_S, L_OUT, TARGET_OUT
    )
except ImportError as e:
    print(f"Error importing from voice_mod.voice: {e}")
    print("Ensure you run this script from the 'dorai' root directory.")
    sys.exit(1)

# Wrapper for Python Logger to ignore ROS kwargs like 'throttle_duration_sec'
class ROSLoggerMock:
    def __init__(self, logger):
        self.logger = logger
    def info(self, msg, **kwargs):
        self.logger.info(msg)
    def warning(self, msg, **kwargs):
        self.logger.warning(msg)
    def error(self, msg, **kwargs):
        self.logger.error(msg)

def _write_wav(filename, audio_float, channels, rate):
    pcm16 = (np.clip(audio_float, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(filename, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm16.tobytes())

def parse_args():
    parser = argparse.ArgumentParser(description="Test Mac Beamformer")
    parser.add_argument("interval", type=str, nargs="?", default="10s", 
                        help="Recording interval (e.g., '10s' or '10')")
    parser.add_argument("--filter", type=str, default="",
                        help="Mic name filter to select specific mics")
    parser.add_argument("--max_mics", type=int, default=3,
                        help="Maximum number of microphones to use")
    return parser.parse_args()

def parse_interval(val):
    if val.endswith('s'):
        return float(val[:-1])
    return float(val)

def _make_callback(ch, logger):
    def callback(indata, frames, time_info, status):
        if status:
            logger.warning(f"{ch.label} status: {status}")
        now = time.monotonic()
        try:
            age = time_info.currentTime - time_info.inputBufferAdcTime
            capture_mono = now - max(age, 0.0)
        except Exception:
            capture_mono = now
        
        # Mix down stereo USB mics (if opened as stereo) to mono
        if indata.shape[1] > 1:
            mono = indata.mean(axis=1)
        else:
            mono = indata[:, 0]
        ch.push(mono.copy(), capture_mono)
    return callback

def main():
    args = parse_args()
    try:
        interval_s = parse_interval(args.interval)
    except ValueError:
        print(f"Invalid interval: {args.interval}")
        sys.exit(1)
    
    # Create directory recording/YYYY-MM-DD-HH-MM-SS/
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    out_dir = os.path.join("recording", timestamp)
    os.makedirs(out_dir, exist_ok=True)
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    base_logger = logging.getLogger("test_beamformer")
    logger = ROSLoggerMock(base_logger)
    
    base_logger.info(f"Will record for {interval_s} seconds into {out_dir}")
    
    # 1. Detect microphones
    # We set include_internal=True so you can test easily on Mac without external mics.
    resolved = detect_microphones(
        logger, include_internal=True, max_mics=args.max_mics,
        requested_rate=48000, prefer_usb=True, capture_channels=0,
        mic_name_filter=args.filter
    )
    
    mics = [
        MicChannel(m["label"], m["device"], m["sample_rate"], m["name"],
                   hpf_hz=150.0, capture_channels=m["capture_channels"])
        for m in resolved
    ]
    num_channels = len(mics)
    
    # 2. Initialize beamformer
    model_path = "dorai_beamformer.ort"
    if not os.path.exists(model_path):
        base_logger.error(f"Model not found at {model_path}. Please run script from 'dorai' root.")
        sys.exit(1)
        
    beam = Beamformer(model_path, num_threads=2, logger=logger)
    beam.warmup(num_channels)
    
    # Continuous enhancer handles whole-signal buffering and overlap-add logic
    enhancer = ContinuousEnhancer(
        beam.run, win=int(2.0 * OUTPUT_RATE), hop=int(1.0 * OUTPUT_RATE), fs=OUTPUT_RATE,
        agc_target_rms=0.06, agc_tc_s=2.0, agc_max_gain=15.0, agc_gate_rms=0.003
    )
    
    # 3. Open PortAudio streams
    for ch in mics:
        blocksize = int(ch.sample_rate * BLOCK_S)
        ch.stream = sd.InputStream(
            samplerate=ch.sample_rate, device=ch.device,
            channels=ch.capture_channels,
            blocksize=blocksize, dtype="float32",
            callback=_make_callback(ch, logger)
        )
        ch.stream.start()
        
    base_logger.info(f"Started recording {num_channels} mic(s) for {interval_s}s...")
    
    raw_blocks = []
    clean_blocks = []
    
    # Worker loop mirroring VoiceMod._tick()
    primed = False
    start_time = time.monotonic()
    next_tick = start_time
    
    try:
        while time.monotonic() - start_time < interval_s:
            next_tick += BLOCK_S
            sleep_t = next_tick - time.monotonic()
            if sleep_t > 0:
                time.sleep(sleep_t)
            else:
                next_tick = time.monotonic()
                
            for ch in mics:
                ch.resample_in()
                
            if len(mics) > 1:
                ref = sum(ch.fifo_len() for ch in mics) / len(mics)
                for ch in mics:
                    ch.update_ratio(ref)
                    
            if not primed:
                depths = [ch.fifo_len() for ch in mics]
                if min(depths) >= max(TARGET_OUT, L_OUT):
                    common = min(depths)
                    for ch in mics:
                        ch.trim_fifo(common)
                    primed = True
                    base_logger.info("FIFOs primed; starting enhancement.")
                else:
                    continue
                    
            min_fifo = min(ch.fifo_len() for ch in mics)
            n_blocks = max(0, (min_fifo - TARGET_OUT) // L_OUT)
            
            for _ in range(int(n_blocks)):
                block = np.stack([ch.pop_block() for ch in mics], axis=0)
                raw_blocks.append(block)
                
                clean = enhancer.push(block)
                if len(clean) > 0:
                    clean_blocks.append(clean)
                    
    except KeyboardInterrupt:
        base_logger.info("Interrupted by user.")
        
    # Flush overlap-add tail
    tail = enhancer.flush()
    if len(tail) > 0:
        clean_blocks.append(tail)
        
    for ch in mics:
        if hasattr(ch, 'stream') and ch.stream is not None:
            ch.stream.stop()
            ch.stream.close()
        
    base_logger.info("Recording finished. Saving files...")
    
    # Save raw mics
    if raw_blocks:
        raw_audio = np.concatenate(raw_blocks, axis=1) # [M, total_samples]
        for i in range(num_channels):
            label = mics[i].label
            fn = os.path.join(out_dir, f"{label}.wav")
            _write_wav(fn, raw_audio[i], channels=1, rate=OUTPUT_RATE)
            base_logger.info(f"Saved {fn}")
            
    # Save clean audio
    if clean_blocks:
        clean_audio = np.concatenate(clean_blocks, axis=0) # [total_samples]
        fn = os.path.join(out_dir, "clean.wav")
        _write_wav(fn, clean_audio, channels=1, rate=OUTPUT_RATE)
        base_logger.info(f"Saved {fn}")

if __name__ == "__main__":
    main()

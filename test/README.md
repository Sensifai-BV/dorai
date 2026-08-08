# Dorai Beamformer Test

`test_beamformer.py` is a standalone Python utility for testing the Dorai multi-microphone beamforming pipeline (`voice_mod`) on any environment where ROS 2 is not installed.

It captures live audio from your microphones using `sounddevice`, passes it through the exact same preprocessing, async-resampling, drift-correction, and `dorai_beamformer.ort` pipeline used in the production ROS 2 node, and saves both the raw multi-channel inputs and the enhanced clean audio as standard `.wav` files for offline analysis.

## Prerequisites

Since this script bypasses ROS 2, it relies on standard Python packages for audio capture and inference. Install them via `pip`:

```bash
pip install numpy scipy sounddevice onnxruntime
```

Make sure that the ONNX model `dorai_beamformer.ort` is present in your working directory (it should already be alongside the script in `test/`).

## Usage

You can run the script via terminal. It takes an optional argument for the recording interval and optional flags to control microphone selection.

```bash
./test_beamformer.py [interval] [options]
```

### Arguments
* `interval` (optional): The duration to record. Defaults to `10s`. You can specify it as `15s` or simply `15`.

### Options
* `--filter <string>`: A case-insensitive substring used to filter and select specific microphone input devices (e.g., `--filter usb` or `--filter rode`).
* `--max_mics <int>`: The maximum number of microphones to capture from (default is 3).

### Examples

Record from any auto-detected microphones for 10 seconds:
```bash
./test_beamformer.py 10s
```

Record for 20 seconds specifically from microphones with "usb" in their name:
```bash
./test_beamformer.py 20s --filter usb
```

## Output

Once the recording interval finishes (or if you manually interrupt it with `Ctrl+C`), the script processes the final audio tail and saves the data into a newly generated timestamped directory:

```text
recording/YYYY-MM-DD-HH-MM-SS/
├── mic0.wav     # Raw capture from the 1st microphone
├── mic1.wav     # Raw capture from the 2nd microphone (if present)
├── mic2.wav     # Raw capture from the 3rd microphone (if present)
└── clean.wav    # The single-channel enhanced output from the beamformer
```

All files are saved at a 16 kHz sample rate. You can drag and drop these `.wav` files into Audacity (or any audio editor) to listen and compare the raw microphone feeds against the beamformer's cleaned output.

# Dorai Acoustic Simulator

## Abstraction
The **Dorai Acoustic Simulator** is a MATLAB-based physical simulation and visualization front-end designed to test the real-time performance of the Dorai speech enhancement model (`dorai_beamformer.ort`). Instead of using actual hardware, the simulator synthesizes realistic multi-channel microphone signals in a virtual environment. It models a **closed acoustic room** (factory environment) containing a human speaker talking to a robotic dog. Sound wave propagation from the speaker's mouth and surrounding industrial noise sources (factory machinery and welding tools) to the microphone array is computed using the **first-order Image Source Method (ISM)**. This models a realistic multipath acoustic space with reflections off the floor, ceiling, and all four walls, allowing the speech-enhancement ONNX model to be benchmarked against room reverberation and spatial noise entirely on a desktop.

![Dorai Acoustic Simulator Scene](acoustic_scene.png)

---

## Specific Details
The physical scene is configured based on the following specific parameters:

### 1. Closed Acoustic Space (Room)
* **Room Dimensions**: $6.0\,\text{m} \times 6.0\,\text{m} \times 3.0\,\text{m}$ (Length $\times$ Width $\times$ Height).
* **Reflection Coefficient**: $0.5$ (modeling wall, floor, and ceiling absorption/reflection losses).
* **Sound Paths**: $7$ distinct propagation paths for each source (1 direct path + 6 first-order boundary reflections).

### 2. Robotic Dog (Receiver Platform)
* **Model**: Artaban robot from Panza.
* **Height**: $50\,\text{cm}$ (the top of its back where the microphone array is mounted is at `z = 0.50 m`).
* **Microphone Array**: A customizable $N$-mic array (2 to 5 channels, default $3$, spacing $10\,\text{cm}$) centered on the dog's back.

### 3. Human Speaker
* **Distance**: $1\,\text{m}$ horizontal distance from the robotic dog.
* **Mouth Height**: $1.70\,\text{m}$ (`z = 1.70 m`).

### 4. Spatial Noise Sources
* **Factory Noise**: Loaded from `wavs/factory.wav` and modeled as a point source located at `[-2.0, 2.0, 1.2]` m in the room.
* **Welding Noise**: Loaded from `wavs/welding.wav` and modeled as a point source located at `[2.0, 2.5, 1.0]` m in the room.

### 5. Signal Processing & Enhancement
* **Sampling Rate**: $16\,\text{kHz}$ (required by the ONNX model).
* **Enhancement Engine**: The noisy microphone array signals (`mic[M, L]`) are processed by the fp32 spatial filter model `voice_mod/dorai_beamformer.ort` to recover single-channel clean speech (`clean[L]`).

---

## How to Run

To run the simulator, execute the setup and launcher script from the `lab/` folder:

```bash
./run_dorai.sh
```

### Script Execution Steps
1. **Base Python Verification**: Checks for Python 3 on your system path.
2. **Virtual Environment**: Creates a local Python virtual environment inside the lab folder at `lab/.pyenv` to keep dependencies self-contained.
3. **Dependency Installation**: Installs the required Python packages (`onnxruntime`, `numpy`, `soundfile`) inside the virtual environment.
4. **MATLAB Launch**: Configures MATLAB's `pyenv` to use the virtual environment interpreter and executes `run_simulation.m` via MATLAB in batch mode, launching the interactive GUI.

### Launch Options
* **Use a specific MATLAB binary**:
  ```bash
  MATLAB_BIN=/Applications/MATLAB_R2025a.app/bin/matlab ./run_dorai.sh
  ```
* **Skip virtual environment setup (after the first run)**:
  ```bash
  SKIP_SETUP=1 ./run_dorai.sh
  ```
* **Run a different MATLAB script**:
  ```bash
  ./run_dorai.sh some_other_script
  ```

# dorai — ROS 2 mic-array speech pipeline
#
# Runs ROS 2 Jazzy in-container, so the host OS is irrelevant: this builds and
# runs identically on Ubuntu 20.04 / 22.04 hosts that only have Docker. The
# same image serves every node (voice_mod, stt_mod, intent_mod, doa_mod);
# docker-compose picks which one each service runs.
FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive

# --- System libraries -------------------------------------------------------
# libportaudio2      : PortAudio backend for `sounddevice` (mic capture)
# libsamplerate0-dev : libsamplerate for the variable-ratio drift resampler
# python3-pip        : to install the pure-Python runtime deps below
# python3-jinja2     : intent_mod renders the GGUF's chat template
# numpy/scipy come from apt so they stay consistent with the apt-managed numpy
# the ROS base image already ships (pip cannot uninstall that debian package).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libportaudio2 \
        libsamplerate0-dev \
        python3-pip \
        python3-numpy \
        python3-scipy \
        python3-jinja2 \
    && rm -rf /var/lib/apt/lists/*

# --- Python runtime dependencies --------------------------------------------
# Not covered by rosdep. --break-system-packages is required because Ubuntu
# 24.04 (Jazzy's base) marks the system Python as externally managed (PEP 668),
# matching the project README. sherpa-onnx runs the Nemotron STT (stt_mod).
# The Nemotron model files (~640 MB each) are NOT baked in: mount ~/dorai/models at
# /root/.cache/dorai/models (see docker-compose.yml) or set DORAI_MODELS_DIR.
RUN pip3 install --no-cache-dir --break-system-packages \
        onnxruntime \
        sounddevice \
        samplerate \
        sherpa-onnx

# intent_mod's LLM runtime. Built from source (no aarch64 wheel): several minutes on a Pi.
# 0.3.36 is the first llama.cpp with Qwen3.5 (Gated DeltaNet) support. The intent GGUF
# (2.6 GB) is NOT baked in: intent_mod downloads it from Hugging Face on first start into
# the mounted ~/dorai/models (see docker-compose.yml).
RUN CMAKE_BUILD_PARALLEL_LEVEL=4 pip3 install --no-cache-dir --break-system-packages llama-cpp-python==0.3.36

# --- Build the ROS 2 workspace ----------------------------------------------
WORKDIR /ros2_ws
COPY voice_mod src/voice_mod
COPY stt_mod   src/stt_mod
COPY doa_mod   src/doa_mod
COPY intent_mod src/intent_mod

# Source the ROS base, then colcon-build the ament_python packages.
RUN . /opt/ros/jazzy/setup.sh && \
    colcon build --symlink-install

ENV DORAI_MODELS_DIR=/root/.cache/dorai/models

# Entrypoint sources both the ROS base and this workspace overlay before exec.
COPY ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh
ENTRYPOINT ["/ros_entrypoint.sh"]

# Default to the capture+beamform node; compose overrides per service.
CMD ["ros2", "run", "voice_mod", "voice"]

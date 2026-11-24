"""Generated signals: commands from text, and industrial noise models."""
from .welding_noise import generate_welding_noise
from .machinery_noise import generate_machinery_noise
from .ventilation_noise import generate_ventilation_noise

NOISE = dict(welding_noise=generate_welding_noise, machinery_noise=generate_machinery_noise,
             ventilation_noise=generate_ventilation_noise)

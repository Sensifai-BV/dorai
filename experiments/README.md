# experiments and evaluation

Run from the repo root. Output goes to `reports/` (not in git): `report.md`,
`results.json`, `results.csv`.

| Script | What it does |
|---|---|
| `experiments/doa_benchmark.py` | DoA on simulated scenes: the doa_mod node against pyroomacoustics SRP-PHAT and MUSIC. Free-field acceptance matrix (8 azimuths), rooms with RT60 0.2-1.0 s, welding noise from -10 to 20 dB. |
| `experiments/locata_doa.py` | doa_mod on real recordings (LOCATA corpus, tasks 1, 3, 5), using the Eigenmike's equatorial capsules as a small planar array. |
| `evaluation/doa_metrics.py` | circular error, MAE, RMSE, median, detection rate, false detections, tracking error |
| `evaluation/doa_node.py` | runs the real doa_mod node over rendered scenes, without ROS, and scores each command's bearing |
| `evaluation/locata.py` | reader for the LOCATA folders: audio, array frame, source bearings over time |

```bash
python -m simulator.acoustic.generators.tts_generator --out ~/dorai/sim/speech                                   # from the repo root
python -m experiments.doa_benchmark --speech ~/dorai/sim/speech              # ~8 min; --quick for a smoke run
python -m experiments.doa_benchmark --speech ~/dorai/sim/speech subtract_noise=true   # any doa_mod parameter
python -m experiments.locata_doa --corpus /path/to/data-corpus [--mics 8]
```

Both run the real node (gate, pooling, publish rule) on `ros2/doa_mod/test/rosmock.py`,
so no ROS is needed. In the benchmark the two baselines are told when each
command starts and ends; the node has to detect the speech itself.

Not here yet: beamforming and speech-enhancement metrics (SI-SDR, PESQ, STOI),
recognition and intent scoring on rendered scenes, other datasets (CHiME-6,
REVERB, LibriCSS), Monte-Carlo runs.

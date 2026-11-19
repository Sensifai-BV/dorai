# Live DoA — direction of arrival from a microphone array

Real-time bearing estimation with a live display. **No geometry input needed**
— the array layout is auto-calibrated from audio at startup.

```
doa/
  auto_calibrate.py   geometry-free calibration (max-TDOA + MDS)
  doa_core.py         estimators (GCC-PHAT, SRP-PHAT, linear + planar solvers)
  live_doa.py         live matplotlib GUI + CLI
  multi_device.py     capture across several independent USB devices
  README.md
```

## Quick start

```bash
python live_doa.py                    # auto-calibrate, then live display
python live_doa.py --list-devices     # find the device index
python live_doa.py --device 7 --fs 0  # specific device, auto sample rate
```

At startup you'll be asked to **walk slowly around the array while speaking**
for ~10 seconds. The system measures pairwise time-delays from many source
positions and reconstructs the mic layout automatically via multidimensional
scaling (MDS). It also auto-detects whether your array is linear or planar.

### How the calibration works

1. You speak while walking around the array, covering as wide an arc as
   possible (ideally a full semicircle or more).
2. For each audio frame, all pairwise GCC-PHAT TDOAs are computed.
3. For each mic pair, the largest observed |TDOA| approximates `d/c` — the
   inter-mic distance divided by the speed of sound.
4. Classical MDS on the pairwise distance matrix yields 2D positions.
5. The system auto-detects linear vs planar geometry.

**Tip:** Cover as wide an arc as possible. A narrow arc underestimates some
mic distances and degrades accuracy. The calibration prints per-pair
"spread" — values above 2× indicate good angular diversity.

### Known circular arrays (shortcut)

If you know your array is circular (e.g. ReSpeaker), you can skip
calibration with `--circle`:

```bash
# ReSpeaker 4-Mic, 32 mm radius
python live_doa.py --device 2 --circle 32

# 5 mics on the rim, 46.3 mm radius
python live_doa.py --device 2 --fs 0 --channels 1 2 3 4 5 --circle 46.3

# 4 on the rim + 1 at the CENTRE
python live_doa.py --device 2 --fs 0 --channels 1 2 3 4 5 --circle 46.3 --centre-mic
```

## What a line can and cannot tell you

If calibration discovers your mics are collinear, it automatically uses the
linear estimator. A linear array measures the angle to its **own axis**:

- `0°` — past the first mic (endfire)
- `90°` — broadside, perpendicular to the line
- `180°` — past the last mic

It **cannot tell which side of the line you are on.** The delay to each mic
depends on `cos(θ)`, and `cos(+40°) = cos(−40°)` exactly. Both branches are
shown — solid and dashed.

**To get a true 360° bearing, move one mic off the line** — even a small
perpendicular offset makes the array planar and resolves the ambiguity. The
calibration will detect this automatically.

## Sample rate

Pass `--fs 0` to use the device's native rate. Most USB mics are 48 kHz and
refuse a 16 kHz open. Higher is better for TDOA: on a 140 mm baseline, 48 kHz
gives 2.92° per sample against 8.77° at 16 kHz, before the 32× interpolation.

## "PortAudio Error: Invalid number of channels"

A configuration mismatch, not a driver fault. The built-in defaults describe a
6-channel ReSpeaker; a mono USB mic exposes one channel. Use
`--device-channels 0` to open at the device's own width.

## Several separate USB microphones

If your mics are separate USB devices:

```bash
python live_doa.py --devices 1 3 4
```

The system will:
1. **Align** the streams (remove start skew)
2. **Auto-calibrate** the geometry
3. Start the live display

**Read this before trusting the number.** Each USB mic has its own crystal.
1 ppm of relative error is 1 µs of drift per second by definition, and cheap
USB audio runs ±50–100 ppm. Simulated on a 3-mic array with a **stationary**
source:

| clocks | bearing sd | total swing over ~10 s |
|---|---|---|
| one shared clock | **0.22°** | **1.19°** |
| independent | 89.2° | **359.19°** |

Trust SHORT-WINDOW bearings and watch the drift readout; for dependable
absolute DoA use one multichannel interface.

**The fix is hardware.** On macOS, build an Aggregate Device (Audio MIDI Setup →
\+ → Create Aggregate Device), tick each mic, and **enable Drift Correction on
every member except the master** — that resamples them onto one clock. Then use
`--device <aggregate idx>` instead of `--devices`.

## The display

- **(a)** array geometry and current bearing; in linear mode the mirror branch
  is drawn dashed
- **(b)** SRP-PHAT spatial spectrum, instantaneous and time-averaged. In linear
  mode the scan covers the half-plane, since a line's spectrum is symmetric and
  a full scan would show every peak twice.
- **(c)** bearing vs time, speech-active frames highlighted

## Method

`doa_linear` solves the single unknown `cos θ` from all pairwise GCC-PHAT
delays by least squares; with three mics the three delays over-determine one
angle, so the residual is a genuine consistency check (large = reverberation, a
wrong spacing, or clock drift). `doa_least_squares` solves the full 2-D
direction vector for non-collinear arrays. `srp_phat` scans bearings and sums
phase-aligned cross-spectra; its whole curve matters, not just the peak, because
a second lobe reveals an interferer the least-squares fit would average in
silently.

## Things that will bite you

1. **Calibrate with enough movement.** Walking a small arc underestimates
   mic distances and produces wrong bearings. The calibration prints warnings
   when angular diversity is low.
2. **The band limit is load-bearing.** Above `c/(2·aperture)` bearings alias.
   Measured on the ReSpeaker, the 2000–2600 Hz band degraded to R = 0.652 while
   neighbouring bands held R ≥ 0.99.
3. **Use one windowing convention.** Comparing unwindowed RMS against a
   windowed-RMS threshold admitted 153 frames where 95 belonged and moved the
   pooled peak by 4°. Do not pre-window a block before `srp_phat` — it windows
   internally, and double-windowing shifted azimuth 1.4°.
4. **Accumulate raw SRP power; normalize once at the end.** Per-frame min-max
   normalization equalizes frames of wildly different SNR and moved the pooled
   peak to +96°.

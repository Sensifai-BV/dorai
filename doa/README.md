# Live DoA — direction of arrival from a microphone array

Real-time bearing estimation with a live display. Python only.

```
doa/
  doa_core.py       estimators (GCC-PHAT, SRP-PHAT, linear + planar solvers)
  live_doa.py       live matplotlib GUI + CLI
  multi_device.py   capture across several independent USB devices
  README.md
```

## Your setup: three mics in a line

That is a **linear array**, and it needs `--linear`, not `--positions`:

```bash
python live_doa.py --device <idx> --device-channels 0 --fs 0 --linear 140 140
```

`--linear` takes the **gaps between consecutive mics in millimetres**, in
channel order — `140 140` means mic1→mic2 is 140 mm and mic2→mic3 is 140 mm.
Measure them; they are the only geometry a line needs, because only distances
*along* the line affect the delays.

Check what device index to use first:

```bash
python live_doa.py --list-devices
```

### Why `--positions 0 0 140 0 70 110` was wrong for you

That described a **triangle** — the third mic offset 110 mm off the line. It
was an example for a 2-D array, and if your mics are actually in a line it
describes geometry you do not have, so every bearing computed from it is wrong.
Geometry is an input, not a detail: the solver trusts it completely.

### What a line can and cannot tell you

A linear array measures the angle to its **own axis**:

- `0°` — past the first mic (endfire)
- `90°` — broadside, perpendicular to the line
- `180°` — past the last mic

It **cannot tell which side of the line you are on.** The delay to each mic
depends on `cos(θ)`, and `cos(+40°) = cos(−40°)` exactly, so a talker 40° above
the line and one 40° below produce byte-identical recordings. This is the *cone
of confusion*. Verified in simulation: true bearings of +35° and −35° return the
identical estimate (35.26°, sd 1.82°). No algorithm removes this — the
information is not in the audio. The display therefore draws **both** branches,
solid and dashed.

If you know which side you are on, `--side +` or `--side -` picks the branch for
display. That is your prior, not a measurement, and it is labelled as such.

**To get a true 360° bearing, move one mic off the line** — even 100 mm of
perpendicular offset makes the array planar and resolves the ambiguity:

```bash
python live_doa.py --device <idx> --device-channels 0 --fs 0 \
                   --positions 0 0  140 0  70 110
```

*(that is where those numbers came from: mic3 offset 110 mm perpendicular.)*

### Where you sit matters: avoid endfire

Sensitivity is not uniform along a line array. Since `τ = (d/c)·cos θ`, the
angular sensitivity `dτ/dθ ∝ sin θ` — it peaks broadside and collapses at the
ends:

| position | µs per degree | a 10 µs error costs |
|---|---|---|
| 90° (broadside, facing the line) | 14.25 | 0.7° |
| 30° | 7.12 | 1.4° |
| 5° (endfire, off the end) | 1.24 | 8.1° |

Broadside is **11.5× more sensitive** than endfire. Sitting off the *end* of the
line is the worst available geometry — sit facing the middle of the array, not
past its end.

### Separate USB mics need a calibration step

Three PortAudio streams start when their own callback first fires — tens of
milliseconds apart. On a 280 mm array the *entire* geometric delay range is
816 µs (39 samples at 48 kHz), so a 20 ms skew is **24× the whole range** and
every frame is physically impossible from the first sample. That is what a
"92% of frames rejected" banner means.

`live_doa.py --devices …` now calibrates at startup and prompts you first.
**Stand in front of the middle of the array, facing it, and speak for 4 s.**

Position matters, and this is the subtle part: a cross-correlation aligner
cannot distinguish start skew from acoustic delay — both are just delay. Run
blind on a *clean* capture of a source at 15°, it measured lags of [0, 19, 38]
samples, which **are** the direction information, and zeroing them moved the
bearing from a correct 15.5° to 90.3° — exactly broadside, the "no delay"
answer. So calibration assumes you are at broadside, where the true acoustic
delay is genuinely zero and any measured lag is purely instrumental.

Measured, with a 20 ms skew injected:

| | frames consistent | reported bearing |
|---|---|---|
| before calibration | 9% | 108° |
| after broadside calibration | 95–100% | 15.2° / 45.1° / 149.9° (true 15/45/150) |

The recovered offsets matched the injected skew exactly. Calibrating from the
wrong spot bakes that angle in as an error on every later bearing.

### Circular array (5 mics, or any N)

A circle resolves the **full 360°** with no mirror ambiguity — the thing a line
cannot do. Two different boards get called "5-mic circle", and they are not
interchangeable:

```bash
# 5 mics on the rim, 72° apart, 46.3 mm radius
python live_doa.py --device 2 --fs 0 --channels 1 2 3 4 5 --circle 46.3

# 4 on the rim + 1 at the CENTRE (the last channel is the centre mic)
python live_doa.py --device 2 --fs 0 --channels 1 2 3 4 5 --circle 46.3 --centre-mic
```

Measure the radius; don't guess it. Validated in simulation at 46.3 mm radius:

| layout | pairs | aperture | aliasing limit | max error |
|---|---|---|---|---|
| 5 on rim | 10 | 88 mm | 1947 Hz | **0.30°** |
| 4 rim + centre | 10 | 93 mm | 1852 Hz | **0.15°** |

over bearings 0/45/90/150/210/300°. The centre mic contributes no delay of its
own — it adds pairs, not aperture.

**Channel-to-position mapping is a wiring convention, not geometry.** `--circle`
knows the mics form a circle; it cannot know which physical mic is channel 1 or
which way the numbering runs. Both are fixable:

```bash
--circle 46.3 30        # mic1 sits at 30° instead of the default 90° (+y)
--clockwise             # channels run clockwise round the rim
```

Get these wrong and every bearing is **rotated or mirrored** by a constant — the
estimate looks confident and is uniformly wrong. Check it against a known
direction before trusting the numbers: speak from a measured angle and compare.
`doa_core.calibrate_offset()` measures the residual delay offsets against a
known bearing if you need to correct for wiring rather than re-map channels.

One caveat for separate USB dongles: **a circle needs all channels on one
clock.** Five independent devices reintroduce the start-skew and drift problems
documented below, which cost far more than the geometry gains. A circular array
is worth it on a single multi-channel device.

### Running with only TWO microphones

Supported, and honest about what it costs:

```bash
python live_doa.py --devices 1 2 --fs 0 --linear 140
```

One gap for two mics. You get the angle to the array axis plus its mirror —
same as any linear array — but you lose the **redundancy** that a third mic
provides, and that loss is not cosmetic:

With three mics the three pairwise delays over-determine one angle, so
disagreement between them is measurable and the residual gate can reject
physically impossible frames. **With two mics there is one pair and one
unknown, so the residual is zero by construction and that gate cannot fire at
all.** Measured on a 2-mic array with 120 ppm drift and a true bearing of 45°,
the residual test alone passed 43% of frames and reported 97.5°.

So a second gate carries the load at M=2: the **peak-to-sidelobe ratio** of the
correlation. A frame dominated by one source has a sharp, isolated peak; drift,
reverberation and noise flatten it. Clean frames sit at PSR ≈ 2.6 (p10 2.56)
where drifting frames fall to p10 0.05. With the PSR floor applied:

| 2-mic case | frames kept | reported |
|---|---|---|
| clean, true 45° | 100% | 45.1° (sd 1.1°) |
| clean, true 120° | 100% | 119.9° (sd 0.9°) |
| 120 ppm drift, true 45° | 10% → **refused** | — |

`psr_min` (default 1.8) is the knob. Raise it in a reverberant room; lower it
only if valid frames are being discarded.

### Refusing bad frames

`doa_linear` reports `consistent`, a physical check: the largest delay the
geometry can produce is `aperture/c` (816 µs for a 280 mm array), so a residual
approaching that means the pairwise delays cannot all come from one plane wave.
Such frames carry no direction.

This matters because averaging them does **not** cancel the error — it pulls the
estimate toward broadside, which then looks like a confident measurement.
Measured with 200 ppm drift: true bearings of 10°, 15° and 30° all reported
85–90°. The display now excludes inconsistent frames, and refuses entirely when
more than half fail, rather than showing a bearing built from the residue.

### Spacing is a real trade-off

Aliasing limit = `c / (2 × aperture)`. Above it, bearings become ambiguous:

| gap (3 mics) | aperture | aliasing limit | usable band |
|---|---|---|---|
| 40 mm | 80 mm | 2144 Hz | 200–2144 Hz |
| 60 mm | 120 mm | 1429 Hz | 200–1429 Hz |
| 80 mm | 160 mm | 1072 Hz | 200–1072 Hz |
| 140 mm | 280 mm | **612 Hz** | 200–612 Hz |
| 200 mm | 400 mm | 429 Hz | 200–429 Hz |

Wider spacing gives finer angular resolution but a lower frequency ceiling.
Speech DoA wants roughly 300–1500 Hz, so **60–80 mm gaps** are the sweet spot
for three mics. At 140 mm gaps the usable band is only 200–612 Hz, which works
but is noisier — the tool prints a warning when the ceiling drops below 800 Hz.

## Sample rate

Pass `--fs 0` to use the device's native rate. Most USB mics are 48 kHz and
refuse a 16 kHz open. Higher is better for TDOA: on a 140 mm baseline, 48 kHz
gives 2.92° per sample against 8.77° at 16 kHz, before the 32× interpolation.

## "PortAudio Error: Invalid number of channels"

A configuration mismatch, not a driver fault. The built-in defaults describe a
6-channel ReSpeaker; a mono USB mic exposes one channel. Use
`--device-channels 0` to open at the device's own width.

## Several separate USB microphones

If your three mics are three separate USB devices rather than one interface:

```bash
python live_doa.py --devices 1 3 4 --linear 140 140
```

**Read this before trusting the number.** Each USB mic has its own crystal.
1 ppm of relative error is 1 µs of drift per second by definition, and cheap USB
audio runs ±50–100 ppm. Simulated on a 3-mic array with a **stationary** source:

| clocks | bearing sd | total swing over ~10 s |
|---|---|---|
| one shared clock | **0.22°** | **1.19°** |
| independent | 89.2° | **359.19°** |

Drift alone rotated the bearing 195° with nothing moving.

Two software corrections were implemented and **both measurably failed**:
low-pass-and-subtract removed the acoustic delay along with the drift (true
bearings 30/90/200° came out 283/309/318°), and line-fitting τ(t) recovered
+22.67 ppm where truth was −80 ppm, because the aperture-bounded search stops
following the ramp. So `DriftMonitor` **measures** drift and reports how long a
bearing stays valid, rather than silently corrupting it, and flags
`DRIFT SATURATED` when the estimate is no longer trustworthy.

Usable window before drift alone moves the bearing 5° at broadside:

| baseline | 1 ppm | 10 ppm | 50 ppm | 100 ppm |
|---|---|---|---|---|
| 60 mm | 15.2 s | 1.5 s | 0.3 s | 0.2 s |
| 100 mm | 25.4 s | 2.5 s | 0.5 s | 0.3 s |
| 250 mm | 63.5 s | 6.4 s | 1.3 s | 0.6 s |

**The fix is hardware.** On macOS, build an Aggregate Device (Audio MIDI Setup →
+ → Create Aggregate Device), tick each mic, and **enable Drift Correction on
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

1. **The band limit is load-bearing.** Above `c/(2·aperture)` bearings alias.
   Measured on the ReSpeaker, the 2000–2600 Hz band degraded to R = 0.652 while
   neighbouring bands held R ≥ 0.99.
2. **Use one windowing convention.** Comparing unwindowed RMS against a
   windowed-RMS threshold admitted 153 frames where 95 belonged and moved the
   pooled peak by 4°. Do not pre-window a block before `srp_phat` — it windows
   internally, and double-windowing shifted azimuth 1.4°.
3. **Accumulate raw SRP power; normalize once at the end.** Per-frame min-max
   normalization equalizes frames of wildly different SNR and moved the pooled
   peak to +96°.
4. **Geometry is an input.** A wrong spacing produces a confident wrong bearing
   with no outward sign except an elevated residual.

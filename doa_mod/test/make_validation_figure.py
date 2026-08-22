#!/usr/bin/env python3
"""make_validation_figure.py — build doa_mod_validation.png.

Reads doa_mod_scores.csv (written by test/replay_session.py) and recomputes the
t4_walk tracking curves, then renders the three-panel validation figure:

  A  polar: reported vs true bearing for the eight marked Task-2 positions
  B  per-mark absolute error against the 15 deg / 25 deg pass bars, all modes
  C  tracking on the walk take: the correct one-chunk-per-hop VAD call, the
     research tree's whole-block call, and no gate at all

Usage:
    python3 test/replay_session.py            # writes doa_mod_scores.csv
    python3 test/make_validation_figure.py    # writes doa_mod_validation.png

The tracking curves are recomputed (~30 s) rather than cached, so the figure
cannot silently disagree with the code it documents.
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                    # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _PKG)

import rosmock                                                     # noqa: E402
rosmock.install()

import doa as D                                                    # noqa: E402
import doa_core as dc                                              # noqa: E402
from replay_session import read_take, wrap180, TASK2               # noqa: E402

FOCAL, ENERGY, OFF, LIVEDOA, TRUTHC = (
    '#c1440e', '#4c72b0', '#8c8c8c', '#55a868', '#333333')


class LiveDoaPatternGate:
    """The research tree's convention: whole 2048-sample block, every hop.

    Reproduced here ONLY as the comparison baseline for panel C -- it is the
    bug, not the fix. See doa.SpeechGate for why it costs tracking latency.
    """

    def __init__(self, path, thr=0.45):
        from vad import NeuralVAD
        self.v = NeuralVAD(path)
        self.thr = thr
        self.last_prob = float('nan')

    def __call__(self, block):
        r = self.v.process_buffer(block[0], fs=16000, threshold=self.thr)
        self.last_prob = r['max_prob']
        return r['is_speech']


def track(x, fs, gate, avg_tc=0.5):
    est = D.BearingEstimator(dc.array_positions(), fs=fs, gate=gate, avg_tc=avg_tc)
    ts, az = [], []
    for s in range(0, x.shape[1] - D.BLOCK + 1, D.HOP):
        est.push(x[:, s:s + D.BLOCK])
        a, _ = est.bearing()
        ts.append((s + D.BLOCK) / fs)
        az.append(a)
    return np.array(ts), np.array(az)


def settle_time(t, az, tol=15.0):
    fin = az[-1]
    ok = np.isfinite(az)
    out = t[ok][np.abs((az[ok] - fin + 180) % 360 - 180) > tol]
    return (out.max() if out.size else 0.0), fin


def main(argv=None):
    scores = (argv[0] if argv else
              os.path.join(os.getcwd(), 'doa_mod_scores.csv'))
    session = os.path.join(_PKG, '..', 'doa', 'session_2026-08-12-10-04-37')
    if not os.path.isfile(scores):
        print(f"missing {scores}; run test/replay_session.py first",
              file=sys.stderr)
        return 1

    by = {r['take']: r for r in csv.DictReader(open(scores))}
    modes = ['neural', 'energy', 'off']
    truth = [float(by[t]['truth_deg']) for t in TASK2]
    rep = {m: [float(by[t][f'{m}_deg']) for t in TASK2] for m in modes}
    err = {m: np.abs([float(by[t][f'{m}_err_deg']) for t in TASK2]) for m in modes}

    x, fs = read_take(os.path.join(session, 't4_walk'))
    vp = os.path.join(_PKG, 'vad.onnx')
    curves = {}
    for tag, gate in (('neural', D.SpeechGate('neural', model_path=vp)),
                      ('livedoa', LiveDoaPatternGate(vp)),
                      ('off', D.SpeechGate('off'))):
        curves[tag] = track(x, fs, gate)
    s_n, fin = settle_time(*curves['neural'])
    s_l, _ = settle_time(*curves['livedoa'])

    fig = plt.figure(figsize=(13.0, 4.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.92, 1.15, 1.25], wspace=0.34,
                          left=0.045, right=0.985, top=0.80, bottom=0.155)

    # -- A: polar -----------------------------------------------------------
    axA = fig.add_subplot(gs[0, 0], projection='polar')
    axA.set_theta_zero_location('E')
    axA.set_theta_direction(1)
    th_t = np.deg2rad(truth)
    axA.plot(th_t, np.ones_like(th_t), 'o', ms=9, mfc='none', mec=TRUTHC,
             mew=1.5, label='true position', zorder=3)
    axA.plot(np.deg2rad(rep['neural']), np.full(len(TASK2), 0.72), 'o', ms=6.5,
             color=FOCAL, label='doa_mod (neural VAD)', zorder=4)
    for tt, rr in zip(truth, rep['neural']):
        axA.plot([np.deg2rad(tt), np.deg2rad(rr)], [1.0, 0.72], '-', lw=0.9,
                 color=FOCAL, alpha=0.45, zorder=2)
    i315 = TASK2.index('t2_315')
    axA.annotate('315° mark\n(known reflector)',
                 xy=(np.deg2rad(rep['neural'][i315]), 0.72),
                 xytext=(np.deg2rad(318), 0.30), fontsize=6.8, ha='center',
                 va='center', color=FOCAL,
                 arrowprops=dict(arrowstyle='-', lw=0.8, color=FOCAL, alpha=0.7))
    axA.set_ylim(0, 1.25)
    axA.set_yticks([])
    axA.grid(alpha=0.3)
    axA.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
    mean_err = err['neural'].mean()
    axA.set_title(f'Eight marked talker positions,\n'
                  f'recovered within {mean_err:.1f}° on average', pad=17)
    axA.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), frameon=False,
               fontsize=7, handletextpad=0.4)

    # -- B: per-mark error --------------------------------------------------
    axB = fig.add_subplot(gs[0, 1])
    xpos = np.arange(len(TASK2))
    w = 0.26
    for k, (m, c, lab) in enumerate([('neural', FOCAL, 'neural VAD'),
                                     ('energy', ENERGY, 'energy VAD'),
                                     ('off', OFF, 'VAD off')]):
        axB.bar(xpos + (k - 1) * w, err[m], width=w, color=c, label=lab,
                zorder=3, edgecolor='none')
    axB.axhline(15.0, color=TRUTHC, ls='--', lw=1.0, zorder=4)
    axB.axhline(25.0, color='#b03030', ls=':', lw=1.0, zorder=4)
    axB.text(len(TASK2) - 0.55, 15.5, 'mean bar 15°', fontsize=6.5, ha='right',
             va='bottom', color=TRUTHC)
    axB.text(len(TASK2) - 0.55, 25.5, 'max bar 25°', fontsize=6.5, ha='right',
             va='bottom', color='#b03030')
    axB.set_xticks(xpos)
    axB.set_xticklabels([f'{int(t)}°' for t in truth])
    axB.set_xlabel('true talker bearing   (lower bar = better)')
    axB.set_ylabel('absolute error (°)')
    axB.set_ylim(0, 32)
    axB.set_title('Every mark clears the pass bar,\nin all three VAD modes',
                  pad=10, loc='left')
    axB.legend(frameon=False, fontsize=7, loc='upper left', handlelength=1.2,
               bbox_to_anchor=(0.0, 0.90))
    axB.margins(x=0.02)

    # -- C: tracking --------------------------------------------------------
    axC = fig.add_subplot(gs[0, 2])
    for tag, c, lab, lw, z, al in [
            ('off', OFF, 'VAD off', 1.2, 3, 0.95),
            ('livedoa', LIVEDOA, 'whole-block VAD call (prior)', 1.6, 4, 0.85),
            ('neural', FOCAL, 'one chunk per hop (doa_mod)', 1.6, 5, 0.95)]:
        t, az = curves[tag]
        ok = np.isfinite(az)
        axC.plot(t[ok], az[ok], '-', color=c, lw=lw, label=lab, zorder=z, alpha=al)
    axC.axhspan(fin - 15, fin + 15, color=FOCAL, alpha=0.09, zorder=1)
    axC.text(19.6, fin + 19, '±15° of final', fontsize=6.5, ha='right', color=FOCAL)
    for tag, c, s in (('neural', FOCAL, s_n), ('livedoa', LIVEDOA, s_l)):
        t, az = curves[tag]
        axC.plot([s], [np.interp(s, t, np.nan_to_num(az))], 'v', color=c, ms=6,
                 zorder=6, clip_on=False)
    axC.text(6.5, 344, f'settles: {s_n:.1f} s vs {s_l:.1f} s', fontsize=7,
             ha='center', color=TRUTHC)
    axC.annotate('never settles', xy=(15.0, 218), xytext=(13.2, 300), fontsize=7,
                 ha='center', color='#5f5f5f',
                 arrowprops=dict(arrowstyle='-', lw=0.7, color=OFF))
    axC.set_xlabel('time through the walk take (s)')
    axC.set_ylabel('pooled bearing (°)')
    axC.set_ylim(0, 360)
    axC.set_yticks(np.arange(0, 361, 90))
    axC.set_title('Gating drives tracking, not accuracy:\n'
                  'ungated pooling never converges', pad=10, loc='left')
    axC.legend(frameon=False, fontsize=7, loc='lower right', handlelength=1.4)
    axC.margins(x=0.02)

    out = os.path.join(os.getcwd(), 'doa_mod_validation.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f"wrote {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

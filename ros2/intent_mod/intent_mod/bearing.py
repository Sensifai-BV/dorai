"""Which talker bearing belongs to a transcript.

doa_mod publishes /dorai_doa only while someone speaks, so its messages come in
bursts, one per utterance. A transcript reaches intent_mod 2-4 s after its
utterance ended (silence that ends the utterance + decoding), and by then the
next talker may already be speaking. The latest bearing is therefore the wrong
one; the right one is the last value of the most recent burst that has ENDED.
The burst still in progress belongs to speech whose transcript has not arrived
yet (stt_mod only emits after the utterance's trailing silence).

The last value of a burst is used because doa_mod's estimate is pooled over the
utterance and is most settled at its end.
"""
import collections


class BearingHistory:
    def __init__(self, gap_s=0.6, max_age_s=20.0, maxlen=2000):
        self.gap_s = float(gap_s)          # silence between messages that ends a burst
        self.max_age_s = float(max_age_s)  # older bursts are not attached to anything
        self._h = collections.deque(maxlen=maxlen)   # (t, bearing_deg), t monotonic seconds

    def add(self, t, bearing):
        self._h.append((float(t), float(bearing)))

    @property
    def latest(self):
        return self._h[-1][1] if self._h else None

    def bursts(self):
        """[(t_start, t_end, last_bearing)], oldest first."""
        out = []
        for t, b in self._h:
            if out and t - out[-1][1] <= self.gap_s:
                out[-1] = (out[-1][0], t, b)
            else:
                out.append((t, t, b))
        return out

    def for_utterance(self, now):
        """Bearing of the most recent finished burst at time ``now``, or None."""
        for t0, t1, b in reversed(self.bursts()):
            if now - t1 <= self.gap_s:
                continue                   # still being spoken: the next utterance
            if now - t1 > self.max_age_s:
                return None
            return b
        return None

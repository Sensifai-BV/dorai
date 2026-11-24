#!/usr/bin/env python3
"""Record every DORAI topic to JSONL (audio summarised, clean audio also saved as WAV)."""
import json, os, sys, time
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray, String

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)

TOPICS = {
    "/dorai_clean_audio": Float32MultiArray,
    "/dorai_raw_audio": Float32MultiArray,
    "/dorai_partial_transcript": String,
    "/dorai_transcript": String,
    "/dorai_stt/status": String,
    "/dorai_stt/metrics": String,
    "/dorai_doa": Float32,
    "/dorai_doa/state": Float32MultiArray,
    "/dorai_doa/human": String,
    "/dorai_stop": String,
    "/dorai_plan": String,
    "/dorai_say": String,
    "/dorai_intent/decision": String,
    "/dorai_intent/status": String,
    "/dorai_sim/truth": String,
}


class Rec(Node):
    def __init__(self):
        super().__init__("dorai_e2e_recorder")
        self.t0 = time.time()
        self.f = open(os.path.join(OUT, "topics.jsonl"), "w")
        self.clean = []
        for t, ty in TOPICS.items():
            self.create_subscription(ty, t, lambda m, t=t: self.cb(t, m), 50)

    def cb(self, topic, m):
        rec = {"t": round(time.time() - self.t0, 3), "topic": topic}
        if isinstance(m, String):
            try:
                rec["data"] = json.loads(m.data)
            except Exception:
                rec["data"] = m.data
        elif isinstance(m, Float32):
            rec["data"] = round(float(m.data), 2)
        else:
            d = np.asarray(m.data, dtype=np.float32)
            if topic.endswith("audio"):
                ch = max(int(d[2]), 1)
                x = d[4:]
                rec["header"] = {"capture_s": round(float(d[0]), 3), "rate": float(d[1]),
                                 "channels": ch, "seq": int(d[3])}
                rec["samples"] = int(x.size)
                rec["duration_s"] = round(x.size / ch / float(d[1]), 3)
                rec["rms_dbfs"] = round(float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)), 1) if x.size else None
                rec["peak"] = round(float(np.max(np.abs(x))), 3) if x.size else None
                if topic == "/dorai_clean_audio":
                    self.clean.append(x.copy())
            else:
                rec["data"] = [round(float(v), 3) for v in d]
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()

    def save(self):
        if self.clean:
            import soundfile as sf
            sf.write(os.path.join(OUT, "clean_audio.wav"), np.concatenate(self.clean), 16000)


def main():
    rclpy.init()
    n = Rec()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    except Exception:
        import traceback; traceback.print_exc()
    finally:
        n.save()
        n.f.close()


if __name__ == "__main__":
    main()

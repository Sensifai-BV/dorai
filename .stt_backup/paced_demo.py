import sys, os, json, time, numpy as np, soundfile as sf
sys.path[:0] = ["test", "."]
import rosmock; rosmock.install()
import stt_mod.node as N
rows = [json.loads(l) for l in open("/Users/javad/External/Projects/stt/datasets/real_mic/javad/manifest.json")]
rows = [r for r in rows if r["condition"] == "quiet" and r["take"] == 2][::2][:12]
segs, truth = [], []
for r in rows:
    x, sr = sf.read(r["audio_filepath"], dtype="float32"); x = x if x.ndim == 1 else x.mean(1)
    segs += [x, np.zeros(int(1.5 * 16000), np.float32)]; truth.append(r["command_id"])
y = np.concatenate(segs + [np.zeros(4 * 16000, np.float32)])
rosmock.Node.overrides = {"models_dir": "/Users/javad/External/Projects/stt/models/onnx", "num_threads": 2}
node = N.SttNode(); cb = node.subs["/dorai_clean_audio"]; fr = 5 * 16000; t0 = time.time()
for k, i in enumerate(range(0, len(y), fr)):
    time.sleep(max(0, t0 + (i + fr) / 16000 - time.time()))      # frame published when its last sample is captured
    cb(rosmock.Float32MultiArray([0.0, 16000, 1, k] + y[i:i + fr].tolist()))
while node._q: time.sleep(0.05)
time.sleep(1.0); node.shutdown(); node._status()
ints = [json.loads(m) for m in node.pubs["/dorai_intent"].msgs]
print("model", node.model_tag, "| audio", round(len(y)/16000,1), "s")
for m in ints: print(f'{m["lag_s"]:5.2f}s  {m["intent"]:10s} {m["score"]:.2f} {"OK " if m["accepted"] else "REJ"}  {m["text"]}')
got = [m["intent"] for m in ints if m["accepted"]]
print("truth   ", truth); print("accepted", got)
print("status", node.pubs["/dorai_stt/status"].msgs[-1])

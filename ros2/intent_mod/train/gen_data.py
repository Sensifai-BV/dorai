#!/usr/bin/env python3
"""Synthetic intent data for the dorai LoRA (text -> plan JSON over schema.TASKS).

Splits (same design as the 2026-10-22 report, section 2):
  train       random synthetic plants; place/object names never overlap dev/test
  dev         the sample shop (intent_mod/config/locations.yaml), training phrasings
  test        a held-out plant (HELDOUT_WORLD) with held-out phrasings (TEST_ONLY templates)
  paraphrase  hand-written natural commands for the sample shop (paraphrase.jsonl, not generated)

Each row: {"world": <name>, "text": <STT-style utterance>, "plan": {"steps": [...]}}
and data/mlx/{train,valid}.jsonl hold the chat form mlx_lm trains on (system prompt = the exact
runtime prompt, schema.system_prompt(locs, objs)).

    python intent_mod/train/gen_data.py --out intent_mod/train/data
"""
import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from intent_mod.schema import load_world, system_prompt, validate  # noqa: E402

NUM = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]

# ---------------------------------------------------------------- world pools
# type -> (aliases, meters). Disjoint from the sample shop and the held-out plant.
TRAIN_PLACES = {
    "furnace": (["furnace", "heat treat furnace", "the oven"], ["temperature"]),
    "hydraulic_press": (["hydraulic press", "the press", "press line"], ["pressure"]),
    "cnc_mill": (["cnc mill", "milling machine", "the mill"], []),
    "lathe": (["lathe", "turning lathe", "the lathe machine"], []),
    "cooling_unit": (["cooling unit", "water chiller", "the chiller"], ["temperature", "flow"]),
    "cutting_table": (["cutting table", "plasma table", "plasma cutter"], []),
    "grinding_bench": (["grinding bench", "grinder station", "the grinding area"], []),
    "storage_rack": (["storage rack", "shelving", "parts shelf"], []),
    "loading_dock": (["loading dock", "shipping door", "the dock door"], []),
    "exit_door": (["exit door", "emergency exit", "the side door"], []),
    "ventilation_fan": (["ventilation fan", "exhaust fan", "fume extractor"], ["speed"]),
    "overhead_crane": (["overhead crane", "the crane", "gantry crane"], ["load"]),
    "office": (["office", "foreman office", "the supervisor desk"], []),
    "break_room": (["break room", "canteen", "kitchen"], []),
    "transformer": (["transformer", "power transformer", "substation"], ["voltage", "temperature"]),
    "water_tank": (["water tank", "the tank", "cooling water tank"], ["level"]),
    "assembly_line": (["assembly line", "assembly station", "the line"], []),
    "packing_station": (["packing station", "packing table", "shipping table"], []),
    "scrap_bin": (["scrap bin", "metal scrap container", "the dumpster"], []),
    "wash_station": (["wash station", "parts washer", "cleaning tank"], ["temperature"]),
    "test_bench": (["test bench", "testing rig", "the test stand"], ["voltage", "current"]),
    "oxygen_tank": (["oxygen tank", "oxygen cylinder", "o two bottle"], ["pressure"]),
    "hvac_unit": (["hvac unit", "air handler", "the ac unit"], ["temperature"]),
    "drill_press": (["drill press", "pillar drill", "the drilling machine"], []),
    "saw": (["band saw", "the saw", "cut off saw"], []),
    "pump_room": (["pump room", "water pump", "the pumps"], ["pressure", "flow"]),
    "paint_line": (["paint line", "coating line", "powder coater"], ["temperature"]),
    "forklift_bay": (["forklift bay", "forklift parking", "truck bay"], []),
    "first_aid": (["first aid station", "medical kit", "safety station"], []),
    "server_rack": (["server rack", "network cabinet", "it closet"], ["temperature"]),
}
WELDER_WORDS = ["welder {t}", "welding device {t}", "welding machine {t}", "welding station {t}",
                "welding cell {t}", "welder number {t}", "weld station {t}"]
TRAIN_OBJECTS = {
    "grinder": ["angle grinder", "grinder"], "clamp": ["clamp", "c clamp"], "gloves": ["gloves", "welding gloves"],
    "tape_measure": ["tape measure", "measuring tape"], "flashlight": ["flashlight", "torch light"],
    "multimeter": ["multimeter", "voltmeter"], "drill": ["drill", "power drill"],
    "screwdriver": ["screwdriver"], "pliers": ["pliers", "vise grip"], "chisel": ["chisel"],
    "level": ["spirit level", "level"], "safety_glasses": ["safety glasses", "goggles"],
    "extension_cord": ["extension cord", "power cable"], "bucket": ["bucket", "pail"],
    "rag": ["rag", "cleaning cloth"], "marker": ["marker", "soapstone"], "wire_brush": ["wire brush"],
    "grease_gun": ["grease gun"], "socket_set": ["socket set", "sockets"], "file": ["metal file", "file"],
    "ladder": ["ladder", "step ladder"], "fire_blanket": ["fire blanket"], "ear_muffs": ["ear muffs", "ear protection"],
    "pipe": ["pipe", "steel pipe"], "bracket": ["bracket", "mounting bracket"], "bolt_box": ["box of bolts", "bolts"],
    "clipboard": ["clipboard", "checklist"], "tablet": ["tablet", "ipad"], "radio": ["radio", "walkie talkie"],
    "torque_wrench": ["torque wrench"],
}
# names used only to make clarify examples (never real places/objects in any world)
UNKNOWN_PLACES = ["basement", "roof", "parking lot", "the warehouse", "reception", "the cafeteria upstairs",
                  "boiler room", "elevator", "the garden", "bathroom", "the lab downstairs", "conference room",
                  "the old shed", "welding station fifty", "the smelter", "cooling tower", "the mezzanine"]
UNKNOWN_OBJECTS = ["bucket of paint", "ladder from outside", "my phone", "the coffee mug", "a pizza", "the laptop",
                   "the blue cylinder", "car keys", "the forklift", "a chair", "the printer", "my jacket",
                   "the red folder", "a sandwich"]

# held-out plant for the test split: names appear in no training world and not in the sample shop
HELDOUT_WORLD = {
    "locations": [
        {"id": "welder_7", "aliases": ["welder seven", "welding device seven", "welding machine seven"],
         "equipment": {"meters": ["current", "voltage"]}},
        {"id": "welder_8", "aliases": ["welder eight", "welding device eight", "welding station eight"],
         "equipment": {"meters": ["current"]}},
        {"id": "welder_k", "aliases": ["welder k", "welding device k", "the tig welder"],
         "equipment": {"meters": ["current", "voltage", "gas_flow"]}},
        {"id": "boiler", "aliases": ["boiler", "steam boiler", "the boiler room"], "equipment": {"meters": ["pressure", "temperature"]}},
        {"id": "generator", "aliases": ["generator", "diesel generator", "backup power"], "equipment": {"meters": ["voltage", "fuel"]}},
        {"id": "robot_cell", "aliases": ["robot cell", "robotic welding cell", "the robot arm cage"], "equipment": {"meters": []}},
        {"id": "quality_lab", "aliases": ["quality lab", "qa room", "measurement room"], "equipment": {"meters": []}},
        {"id": "paint_booth", "aliases": ["paint booth", "spray booth"], "equipment": {"meters": ["temperature"]}},
        {"id": "material_store", "aliases": ["material store", "steel stock", "raw material rack"], "equipment": {"meters": []}},
        {"id": "base", "aliases": ["base", "home", "home position", "start point"], "equipment": {"meters": []}},
        {"id": "charging_station", "aliases": ["charging station", "charger", "dock", "charging dock"], "equipment": {"meters": []}},
    ],
    "objects": [
        {"id": "hacksaw", "aliases": ["hacksaw", "the hacksaw"]}, {"id": "mallet", "aliases": ["mallet", "rubber mallet"]},
        {"id": "face_shield", "aliases": ["face shield", "visor"]}, {"id": "caliper", "aliases": ["caliper", "vernier"]},
        {"id": "spool", "aliases": ["wire spool", "spool of wire"]}, {"id": "mig_torch", "aliases": ["mig torch", "mig gun"]},
    ],
}
BASE = {"id": "base", "aliases": ["base", "home", "home position", "start point"], "equipment": {"meters": []}}
CHARGER = {"id": "charging_station", "aliases": ["charging station", "charger", "dock", "charging dock"],
           "equipment": {"meters": []}}


def random_world(rng):
    """8-14 places (2-4 welders + other types + base + charger), 4-8 objects."""
    locs = []
    tags = rng.sample(list("abcdefghjmnpqrstuvwyz") + [str(i) for i in range(1, 13)], rng.randint(2, 4))
    for t in tags:
        spoken = NUM[int(t)] if t.isdigit() else t
        al = rng.sample(WELDER_WORDS, rng.randint(2, 4))
        meters = rng.choice([["current", "voltage"], ["current"], ["current", "voltage", "gas_flow"], ["voltage"]])
        locs.append({"id": f"welder_{t}", "aliases": [a.format(t=spoken) for a in al], "equipment": {"meters": meters}})
    for k in rng.sample(list(TRAIN_PLACES), rng.randint(4, 8)):
        al, meters = TRAIN_PLACES[k]
        locs.append({"id": k, "aliases": rng.sample(al, rng.randint(2, len(al))), "equipment": {"meters": list(meters)}})
    rng.shuffle(locs)
    locs += [BASE, CHARGER]
    objs = [{"id": k, "aliases": list(TRAIN_OBJECTS[k])} for k in rng.sample(list(TRAIN_OBJECTS), rng.randint(4, 8))]
    return ({l["id"]: l for l in locs}, {o["id"]: o for o in objs})


# ---------------------------------------------------------------- phrasing
# Each list: templates used for train/dev; TEST_ONLY[...] are held out for the test split.
T = {
    "goto": ["go to {P}", "head over to {P}", "drive to {P}", "move to {P}", "navigate to {P}", "get yourself to {P}",
             "can you go to {P}", "walk over to {P}", "i need you at {P}", "go over by {P}", "head to {P} please",
             "go stand next to {P}", "make your way to {P}", "please go to {P} now", "go to {P} and wait there"],
    "inspect_meter": ["go to {P} and check the meter", "go to {P} for check meter", "check the meter on {P}",
                      "read the gauge at {P}", "what does the meter on {P} say", "check the {M} on {P}",
                      "tell me the {M} on {P}", "take a reading at {P}", "go read the gauges over at {P}",
                      "check {P} meter", "go to {P} and read the {M}", "read the meter at {P} for me",
                      "check the gauges at {P}", "what is the {M} reading at {P}"],
    "inspect_visual": ["inspect {P}", "go look at {P}", "check {P} for damage", "see if anything is wrong at {P}",
                       "have a look around {P}", "take a picture of {P}", "start inspection at {P}",
                       "do a visual check of {P}", "go inspect {P}", "look over {P} for problems"],
    "read_meter": ["check the meter", "read the meter", "read the gauge", "what does the gauge say", "what's the reading",
                   "tell me what the meter says", "check this meter", "read this gauge for me", "check the gauge here",
                   "what is the meter showing", "read the dial", "read the display"],
    "inspection": {"start": ["start inspection", "start the inspection", "begin the inspection round", "start your rounds",
                             "do your inspection round", "start the patrol", "begin inspection", "resume the inspection"],
                   "stop": ["stop the inspection", "end inspection", "finish your rounds", "stop patrolling",
                            "end the inspection round", "pause the inspection"]},
    "look_at": ["look at {O}", "point your camera at {O}", "show me {O}", "take a look at {O}", "can you see {O}",
                "focus your camera on {O}", "look at {O} please", "have a look at {O}"],
    "dock": ["go charge", "go to the charging station", "dock", "go dock yourself", "you need a recharge go dock",
             "go to the charger", "plug yourself in", "go recharge your battery", "battery is low go charge",
             "head to the charging dock", "go charge yourself up"],
    "come_here": ["come here", "come to me", "come over here", "get over here", "come closer", "come to where i am",
                  "walk to me", "over here please", "come here please", "come to me now"],
    "follow_me": ["follow me", "come with me", "walk with me", "stay behind me", "keep following me", "follow me please",
                  "tag along with me"],
    "wait": ["wait here", "stay", "stay here", "hold your position", "stay put", "wait", "don't move", "wait for me here"],
    "posture": {"sit": ["sit down", "sit", "take a seat", "sit down please", "go sit"],
                "stand": ["stand up", "get up", "stand", "stand up please"],
                "lie_down": ["lie down", "lay down", "go lie down", "lie flat"]},
    "go_back": ["go back", "go back to where you were", "return to your previous spot", "go back where you came from",
                "back to the last place", "go back please"],
    "goto_base": ["go back to base", "return to base", "go home", "head home", "go to base", "back to your home position",
                  "return home", "go back home"],
    "turn": {"left": ["turn left", "turn to the left", "rotate left", "face left", "swing left", "turn left please"],
             "right": ["turn right", "turn to the right", "rotate right", "face right", "swing right", "turn right please"],
             "around": ["turn around", "spin around", "do a u turn", "face the other way", "turn back around"]},
    "move": {"forward": ["move forward {N} meters", "go ahead {N} meters", "drive forward {N} meters", "roll ahead {N} meters",
                         "go forward {N} meters"],
             "backward": ["back up {N} meters", "reverse {N} meters", "move back {N} meters", "step back {N} meters",
                          "drive backward {N} meters"]},
    "set_speed": {"slow": ["slow down", "go slower", "reduce your speed", "slow down please"],
                  "fast": ["speed up", "go faster", "hurry up a bit", "move quicker"],
                  "normal": ["normal speed", "back to normal speed", "go at normal speed"]},
    "weld": {"start": ["start welding", "begin the weld", "strike an arc", "start the weld"],
             "stop": ["stop welding", "end the weld", "finish welding", "stop the weld now"]},
    "gripper": {"open": ["open the gripper", "release it", "let go", "open your hand"],
                "close": ["close the gripper", "grip it", "close your hand", "close the claw"]},
    "arm": {"lift": ["lift the arm", "raise your arm", "arm up", "lift your arm up"],
            "lower": ["lower the arm", "put your arm down", "arm down", "lower your arm"]},
    "pick": ["pick up {O}", "grab {O}", "take {O}", "pick {O} up"],
    "bring": ["bring me {O}", "fetch {O}", "get me {O}", "hand me {O}", "carry {O} over to me", "bring {O} here"],
    "hold": ["hold {O}", "hold on to {O}", "keep holding {O}"],
    "report": {"battery": ["what is your battery level", "how much battery do you have", "battery status",
                           "how charged are you", "tell me your battery"],
               "status": ["report status", "status report", "how are you doing", "are you ok", "give me a status update"]},
    "lights": {"on": ["lights on", "turn on the lights", "switch your lights on", "turn your lights on"],
               "off": ["lights off", "turn off the lights", "switch your lights off", "kill the lights"]},
    "cancel": ["cancel", "cancel that", "never mind", "forget it", "abort the task", "scratch that", "cancel the last task"],
    "confirm": {"yes": ["yes", "yes confirm", "correct", "that's right", "yeah do it", "affirmative"],
                "no": ["no", "no don't", "negative", "nope", "no that's wrong"]},
    "unclear": ["go to the", "uh go over by the", "do the thing", "that one", "check", "go", "can you um",
                "take it over there", "move it there", "the uh", "do it", "over there"],
    "none": ["the forklift needs new tyres", "did you see the game last night", "lunch is at twelve",
             "who left the door open", "my shift ends at five", "this coffee is terrible", "we need more argon next week",
             "the boss is coming tomorrow", "can somebody call maintenance", "i think it's going to rain",
             "where did i put my keys", "that weld looks good to me", "the new guy starts monday",
             "turn up the radio", "is it break time yet"],
}
TEST_ONLY = {
    "goto": ["could you roll over to {P}", "proceed to {P}", "i want you over at {P}", "travel to {P}"],
    "inspect_meter": ["how many {M} is {P} showing", "give me the {M} at {P}", "swing by {P} and read the gauges",
                      "what's {P} reading on the meter"],
    "inspect_visual": ["examine {P}", "go and eyeball {P}", "check out {P} visually", "is anything broken at {P}"],
    "read_meter": ["what number is on the meter", "read out that gauge", "what's the gauge at"],
    "look_at": ["point at {O} with your camera", "look over at {O}", "aim your camera at {O}"],
    "bring": ["could you bring {O} over", "i need {O} bring it", "deliver {O} to me"],
    "come_here": ["get over to me", "come over to where i'm standing", "approach me"],
    "follow_me": ["stick with me", "walk behind me", "follow along"],
    "dock": ["time to charge go dock", "go top up your battery", "return to the charger"],
    "goto_base": ["head back to base", "go back to your starting point", "return to home position"],
    "none": ["the welding fumes are bad today", "i'll be back after lunch", "nice job on that seam",
             "the shipment is late again"],
}
MULTI_CONNECT = [" and then ", " then ", " and ", " and after that ", " after that "]
METER_WORDS = {"current": ["current", "amps", "amperage"], "voltage": ["voltage", "volts"], "gas_flow": ["gas flow"],
               "pressure": ["pressure"], "flow": ["flow", "flow rate"], "temperature": ["temperature", "temp"],
               "speed": ["speed"], "load": ["load"], "level": ["level"], "fuel": ["fuel level", "fuel"]}
STT_ERR = [("welding", "wedding"), ("welding", "welling"), ("meter", "meeter"), ("meter", "metre"), ("gauge", "gage"),
           ("check", "chuck"), ("charging", "charting"), ("lights", "lice"), ("reading", "ready"), ("torch", "porch"),
           ("follow", "fallow"), ("inspection", "inspections"), ("station", "stations"), ("sit", "set")]
FILLERS = ["uh ", "um ", "okay ", "please ", "hey ", "now ", "so ", "can you ", "could you please ", "i need you to "]


class Gen:
    def __init__(self, rng, test_phrasing=False):
        self.rng, self.test = rng, test_phrasing

    def pick(self, key, sub=None):
        base = T[key][sub] if sub else T[key]
        pool = (TEST_ONLY.get(key, []) if self.test and not sub else []) or base
        if self.test and not sub and key in TEST_ONLY and self.rng.random() < 0.3:
            pool = base                                  # test mixes held-out and seen phrasings
        return self.rng.choice(pool)

    def alias(self, ent):
        a = self.rng.choice(ent.get("aliases") or [ent["id"].replace("_", " ")])
        if self.rng.random() < 0.15:                     # STT sometimes writes digits
            for i, w in enumerate(NUM):
                a = a.replace(f" {w}", f" {i}") if a.endswith(f" {w}") else a
        if self.rng.random() < 0.35 and not a.startswith("the "):
            a = "the " + a
        return a

    def step(self, locs, objs, task):
        """-> (text, step dict) for one actionable step, or None if this world cannot express it."""
        r = self.rng
        meters = [i for i, l in locs.items() if l.get("equipment", {}).get("meters")]
        places = [i for i in locs if i not in ("base", "charging_station")]
        if task == "goto":
            p = r.choice(places)
            return self.pick("goto").format(P=self.alias(locs[p])), {"task": "goto", "target": p}
        if task == "goto_base":
            return self.pick("goto_base"), {"task": "goto", "target": "base"}
        if task == "inspect_meter" and meters:
            p = r.choice(meters)
            m = r.choice(METER_WORDS.get(r.choice(locs[p]["equipment"]["meters"]), ["meter"]))
            return self.pick("inspect_meter").format(P=self.alias(locs[p]), M=m), {"task": "inspect_meter", "target": p}
        if task == "inspect_visual":
            p = r.choice(places)
            return self.pick("inspect_visual").format(P=self.alias(locs[p])), {"task": "inspect_visual", "target": p}
        if task in ("look_at", "pick", "bring", "hold") and objs:
            o = r.choice(list(objs))
            return self.pick(task).format(O=self.alias(objs[o])), {"task": task, "object": o}
        if task == "move":
            d = r.choice(["forward", "backward"])
            n = r.choice([1, 2, 3, 4, 5, 0.5, 1.5, 10])
            spoken = {0.5: "half a", 1.5: "one and a half"}.get(n) or (NUM[n] if r.random() < 0.8 else str(n))
            text = self.pick("move", d).format(N=spoken)
            if n == 1:
                text = text.replace("meters", "meter")
            return text, {"task": "move", "direction": d, "meters": n}
        argd = {"inspection": "state", "posture": "pose", "turn": "direction", "set_speed": "speed", "weld": "state",
                "gripper": "state", "arm": "state", "report": "what", "lights": "state", "confirm": "answer"}
        if task in argd:
            v = r.choice(list(T[task]))
            return self.pick(task, v), {"task": task, argd[task]: v}
        if task in ("read_meter", "dock", "come_here", "follow_me", "wait", "go_back", "cancel"):
            return self.pick(task), {"task": task}
        return None

    def clarify(self, locs, objs):
        r = self.rng
        kind = r.choice(["unknown_target"] * 3 + ["unknown_object"] * 2 + ["unclear"] * 2)
        if kind == "unknown_target":
            u = r.choice([n for n in UNKNOWN_PLACES if not any(n.replace("the ", "") in a for l in locs.values()
                                                             for a in l.get("aliases", []))])
            key = r.choice(["goto", "inspect_meter", "inspect_visual"])
            t = self.pick(key).format(P=u, M=r.choice(["pressure", "meter", "temperature"]))
        elif kind == "unknown_object":
            u = r.choice(UNKNOWN_OBJECTS)
            t = self.pick(r.choice(["bring", "pick", "look_at"])).format(O=u)
        else:
            t = r.choice(T["unclear"])
        return t, {"steps": [{"task": "clarify", "reason": kind}]}

    def utterance(self, locs, objs):
        r = self.rng
        x = r.random()
        if x < 0.10:
            text, plan = self.clarify(locs, objs)
        elif x < 0.17:
            text, plan = self.pick("none"), {"steps": [{"task": "none"}]}
        else:
            n = 1 if r.random() < 0.82 else r.choice([2, 2, 2, 3])
            weights = {"goto": 5, "goto_base": 3, "inspect_meter": 7, "inspect_visual": 4, "read_meter": 4,
                       "inspection": 3, "look_at": 4, "dock": 3, "come_here": 3, "follow_me": 3, "wait": 2,
                       "posture": 3, "go_back": 2, "turn": 3, "move": 3, "set_speed": 2, "weld": 2, "gripper": 1,
                       "arm": 1, "pick": 2, "bring": 3, "hold": 1, "report": 2, "lights": 1, "cancel": 1, "confirm": 1}
            if n > 1:                                    # dialog acts do not chain
                for k in ("cancel", "confirm"):
                    weights.pop(k)
            texts, steps = [], []
            while len(steps) < n:
                task = r.choices(list(weights), list(weights.values()))[0]
                s = self.step(locs, objs, task)
                if s and s[1] not in steps:
                    texts.append(s[0])
                    steps.append(s[1])
            text = texts[0] + "".join(r.choice(MULTI_CONNECT) + t for t in texts[1:])
            plan = {"steps": steps}
        return self.noise(text), plan

    def noise(self, t):
        r = self.rng
        if r.random() < 0.25:
            t = r.choice(FILLERS) + t
        if r.random() < 0.15:
            a, b = r.choice(STT_ERR)
            t = t.replace(a, b, 1)
        if r.random() < 0.10:
            t = t.replace("the ", "", 1)
        if r.random() < 0.04:
            t = r.choice(["artaban ", "hey artaban "]) + t     # wake word the gate left in
        if r.random() < 0.30:                                   # cased + punctuated STT output
            t = t[0].upper() + t[1:] + r.choice([".", ".", "?", "!"])
        return t


def chat_row(locs, objs, text, plan):
    return {"messages": [{"role": "system", "content": system_prompt(locs, objs)},
                         {"role": "user", "content": text},
                         {"role": "assistant", "content": json.dumps(plan, separators=(",", ":"))}]}


def build(gen, worlds, n, name):
    rows, seen = [], set()
    tries = 0
    while len(rows) < n and tries < n * 50:
        tries += 1
        wname, (locs, objs) = worlds[len(rows) % len(worlds)] if len(worlds) > 1 else worlds[0]
        text, plan = gen.utterance(locs, objs)
        if (wname, text) in seen:
            continue
        errs = [e for e in validate(plan, locs, objs)]
        assert not errs, (text, plan, errs)
        seen.add((wname, text))
        rows.append({"world": wname, "text": text, "plan": plan})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "data"))
    ap.add_argument("--n-train", type=int, default=1200)
    ap.add_argument("--n-worlds", type=int, default=80)
    ap.add_argument("--n-dev", type=int, default=150)
    ap.add_argument("--n-test", type=int, default=150)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)

    train_worlds = [(f"synth_{i:03d}", random_world(rng)) for i in range(a.n_worlds)]
    shop = load_world(os.path.join(os.path.dirname(HERE), "config", "locations.yaml"))
    held = ({l["id"]: l for l in HELDOUT_WORLD["locations"]}, {o["id"]: o for o in HELDOUT_WORLD["objects"]})

    train = build(Gen(rng), train_worlds, a.n_train, "train")
    dev = build(Gen(rng), [("sample_shop", shop)], a.n_dev, "dev")
    test = build(Gen(rng, test_phrasing=True), [("heldout_plant", held)], a.n_test, "test")
    # no test/dev sentence may appear in train (same text, any world)
    tr_text = {r["text"].lower().strip(".?!") for r in train}
    test = [r for r in test if r["text"].lower().strip(".?!") not in tr_text]

    wmap = dict(train_worlds)
    with open(os.path.join(a.out, "worlds.json"), "w") as f:
        json.dump({k: {"locations": list(v[0].values()), "objects": list(v[1].values())} for k, v in
                   list(wmap.items()) + [("heldout_plant", held)]}, f)
    for name, rows in (("train_rows", train), ("dev", dev), ("test", test)):
        with open(os.path.join(a.out, f"{name}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    # mlx_lm chat format (own dir: mlx_lm would read test.jsonl too): train on synthetic plants,
    # valid = sample shop (dev), system prompt included
    os.makedirs(os.path.join(a.out, "mlx"), exist_ok=True)
    with open(os.path.join(a.out, "mlx", "train.jsonl"), "w") as f:
        for r in train:
            f.write(json.dumps(chat_row(*wmap[r["world"]], r["text"], r["plan"])) + "\n")
    with open(os.path.join(a.out, "mlx", "valid.jsonl"), "w") as f:
        for r in dev[:60]:
            f.write(json.dumps(chat_row(*shop, r["text"], r["plan"])) + "\n")
    from collections import Counter
    c = Counter(s["task"] for r in train for s in r["plan"]["steps"])
    print(f"train {len(train)}  dev {len(dev)}  test {len(test)}  worlds {len(train_worlds)}")
    print("train task counts:", dict(sorted(c.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()

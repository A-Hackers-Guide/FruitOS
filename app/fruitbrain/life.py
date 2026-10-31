"""The always-on fly. The brain runs continuously and is in control: what the fly does is read
only from its descending command neurons, the cells that carry orders from brain to body.

Command neurons used, each with the experiment that showed what it does:
  DNp01 "Giant Fiber"    -> escape jump / take-off      (von Reyn et al. 2014, Nat Neurosci)
  MDN "moonwalker"       -> walk backward               (Bidaye et al. 2014, Science)
  DNa01, DNa02           -> steer left / right          (Rayshubskiy et al. 2020; Yang et al. 2023)
  DNp09                  -> walk forward                (Bidaye et al. 2020, Neuron)
"""
import os
import threading
import time
from collections import deque

import numpy as np

from .brain import SENSES

CHUNK_MS = 100.0          # brain time simulated per step
WINDOW = 3                # behaviour is judged over the last WINDOW chunks
NOISE_HZ = 1.0            # background trickle into every sensory neuron (the world is never silent)
IDLE_PAUSE_S = 60.0       # nobody watching -> the fly rests and the CPU idles
CPU_SHARE = float(os.environ.get("FRUITOS_CPU_SHARE", "0.5"))   # fraction of one core to use

BEHAVIOURS = [
    # name, command neuron types, what the fly does, Hz needed, evidence
    ("escape", ["DNp01"], "jump and take off", 20.0, "von Reyn et al. 2014"),
    ("backward", ["MDN"], "walk backward", 20.0, "Bidaye et al. 2014"),
    ("turn", ["DNa01", "DNa02"], "turn", 20.0, "Rayshubskiy et al. 2020; Yang et al. 2023"),
    ("forward", ["DNp09"], "walk forward", 20.0, "Bidaye et al. 2020"),
]

STIMULI = {
    # sense -> (cell types to drive, what the learner is doing, max neurons)
    "smell": (["ORN_DA1"], "a puff of fly scent (the DA1 smell channel)", 100),
    "taste": (["claw_tpGRN"], "something touching the taste hairs on my feet", 60),
    "loom": (["LPLC2", "LC4"], "a dark shadow rushing toward me", 150),
    "hearing": (None, "a buzzing sound near my antennae", 100),
    "wind": (None, "a gust of wind", 100),
    "touch": (["SNta29"], "a touch on my body hairs", 100),
    "heat": (None, "a warm spot", 25),
    "humidity": (None, "damp air", 66),
}


class Life:
    def __init__(self, brain, seed=0):
        self.b = brain
        self.state = brain.new_state(seed)
        self.lock = threading.Lock()
        sensory = [k for k, v in enumerate(brain.vocab["superclass"]) if "sensory" in v]
        self.noise = np.flatnonzero(np.isin(brain.superclass, sensory))
        self.groups = []
        for name, types, does, hz, cite in BEHAVIOURS:
            idx = np.array(sorted(i for t in types for i in brain.neurons_of_type(t)), dtype=np.int64)
            self.groups.append({"name": name, "types": types, "does": does, "hz": hz, "cite": cite, "idx": idx})
        self.stim_sets = {}
        for sense, (types, _, cap) in STIMULI.items():
            if types:
                idx = [i for t in types for i in brain.neurons_of_type(t)]
            else:
                idx = brain.sense_neurons(sense).tolist()
            self.stim_sets[sense] = np.array(idx[:cap], dtype=np.int64)
        self.recent = deque(maxlen=WINDOW)
        self.queue = []            # [sense, remaining_ms]
        self.behaviour = "rest"
        self.turn_side = None
        self.because = None        # sense that was being felt when the behaviour started
        self.history = deque(maxlen=40)
        self.last_active = np.zeros(0, dtype=np.int64)
        self.last_counts = np.zeros(brain.n, dtype=np.int32)
        self.brain_ms = 0.0
        self.seq = 0
        self.last_viewer = time.time()
        self._stop = False
        self.thread = threading.Thread(target=self._loop, daemon=True)

    # ------------------------------------------------------------ control
    def start(self):
        self.thread.start()

    def stop(self):
        self._stop = True

    def poke(self, sense, ms=300.0):
        with self.lock:
            self.queue.append([sense, ms])
        self.last_viewer = time.time()

    def heartbeat(self):
        self.last_viewer = time.time()

    # ------------------------------------------------------------ the loop
    def _loop(self):
        while not self._stop:
            if time.time() - self.last_viewer > IDLE_PAUSE_S and not self.queue:
                time.sleep(0.5)
                continue
            t0 = time.time()
            self.step()
            busy = time.time() - t0
            time.sleep(busy * (1 - CPU_SHARE) / max(CPU_SHARE, 0.05))

    def step(self):
        with self.lock:
            sense = self.queue[0][0] if self.queue else None
            if self.queue:
                self.queue[0][1] -= CHUNK_MS
                if self.queue[0][1] <= 0:
                    self.queue.pop(0)
        stim = self.stim_sets.get(sense, np.zeros(0, dtype=np.int64))
        with self.b._lock:
            out = self.state.run(CHUNK_MS, stim, noise=self.noise, noise_hz=NOISE_HZ, frame_ms=CHUNK_MS)
        self.recent.append(out["counts"])
        self.brain_ms += CHUNK_MS
        self.last_counts = out["counts"]
        self.last_active = np.flatnonzero(out["counts"])
        self._decide(sense)
        self.seq += 1

    def _decide(self, sense):
        """The brain decides: the first command group firing above its threshold wins."""
        total = np.sum(self.recent, axis=0)
        secs = len(self.recent) * CHUNK_MS / 1000.0
        chosen, side, rates = "rest", None, {}
        for grp in self.groups:
            if grp["idx"].size == 0:
                continue
            r = total[grp["idx"]] / secs
            rates[grp["name"]] = float(r.mean())
            if chosen == "rest" and r.mean() >= grp["hz"]:
                chosen = grp["name"]
                if grp["name"] == "turn":
                    sides = [self.b.label("side", i) for i in grp["idx"]]
                    left = sum(r[k] for k, s in enumerate(sides) if s == "L")
                    right = sum(r[k] for k, s in enumerate(sides) if s == "R")
                    side = "left" if left > right else "right"
        self.rates = rates
        # every other command neuron the brain is using right now, known job or not
        dn = self.b.mask("superclass", ["descending_neuron"])
        r_all = total / secs
        cand = np.flatnonzero(dn & (r_all >= 10))
        cand = cand[np.argsort(r_all[cand])[::-1]][:6]
        self.commands = [{"index": int(i), "type": self.b.type[i] or "(untyped)",
                          "rate_hz": round(float(r_all[i]), 1)} for i in cand]
        if chosen != self.behaviour:
            self.history.append({"brain_ms": self.brain_ms, "from": self.behaviour, "to": chosen,
                                 "side": side, "sense": sense, "rates": rates})
            self.because = sense
        self.behaviour, self.turn_side = chosen, side

    # ------------------------------------------------------------ reporting
    def group(self, name):
        return next(g for g in self.groups if g["name"] == name)

    def snapshot(self, max_points=4000):
        active = self.last_active
        if active.size > max_points:
            active = np.sort(np.random.default_rng(self.seq).choice(active, max_points, replace=False))
        return {
            "seq": self.seq, "brain_ms": self.brain_ms, "behaviour": self.behaviour,
            "turn_side": self.turn_side, "because": self.because,
            "feeling": self.queue[0][0] if self.queue else None,
            "active_count": int(self.last_active.size), "active": active.tolist(),
            "rates": {k: round(v, 1) for k, v in getattr(self, "rates", {}).items()},
            "commands": getattr(self, "commands", []),
            "paused": time.time() - self.last_viewer > IDLE_PAUSE_S and not self.queue,
        }

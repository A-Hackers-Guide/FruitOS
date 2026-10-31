"""The fly's brain: the real MaleCNS wiring diagram driving a leaky integrate-and-fire simulation.

Model after Shiu et al. 2024 (Nature, "A Drosophila computational brain model reveals
sensorimotor processing"): every traced neuron is a leaky integrate-and-fire unit, every synapse
adds a fixed voltage kick, and the sign comes from the neuron's predicted transmitter.
CPU only, single-threaded numpy; cost scales with the number of spikes, not with connections.
"""
import json
import os
import threading

import numpy as np

# Shiu et al. 2024 parameters
V_REST = -52.0      # mV
V_RESET = -52.0
V_TH = -45.0
TAU_M = 20.0        # ms
TAU_SYN = 5.0       # ms
T_REF = 2.2         # ms
W_SYN = 0.15        # mV per synapse (Shiu: 0.275 on FlyWire; recalibrated for the MaleCNS)
DELAY = 1.8         # ms
DT = 0.1            # ms
STIM_HZ = 150.0     # Poisson drive to stimulated neurons
# Spike-frequency adaptation (not in Shiu et al.): each spike raises that neuron's threshold,
# which decays back. Real neurons do this; without it this larger CNS dataset runs away.
ADAPT_MV = 4.0
TAU_ADAPT = 100.0   # ms
MAX_FRAME_NEURONS = 8000

SENSES = {
    "smell": {"cls": ["olfactory"], "say": "smell"},
    "taste": {"cls": ["gustatory"], "say": "taste"},
    "sight": {"cls": ["visual"], "say": "see light"},
    "touch": {"cls": ["mechanosensory_tactile"], "say": "feel a touch"},
    "hearing": {"subclass": ["auditory"], "say": "hear a sound"},
    "wind": {"subclass": ["wind_gravity"], "say": "feel wind and gravity"},
    "heat": {"cls": ["thermosensory"], "say": "feel temperature change"},
    "humidity": {"cls": ["hygrosensory"], "say": "sense humidity"},
    "body": {"cls": ["mechanosensory_proprioceptive"], "say": "feel my own body move"},
}


def default_data_dir():
    return os.environ.get("FRUITOS_DATA", "/opt/fruitos/data")


class Brain:
    def __init__(self, data_dir=None):
        d = data_dir or default_data_dir()
        z = np.load(os.path.join(d, "brain.npz"))
        self.indptr = z["indptr"]
        self.indices = z["indices"]
        self.weight = z["weight"]
        self.sign = z["sign"]
        self.pos = z["pos"]
        self.measured = z["measured"]
        self.superclass = z["superclass"]
        self.cls = z["cls"]
        self.subclass = z["subclass"]
        self.nt = z["nt"]
        self.side = z["side"]
        with open(os.path.join(d, "meta.json")) as f:
            meta = json.load(f)
        self.vocab = meta["vocab"]
        self.body_id = np.array(meta["body_id"], dtype=np.int64)
        self.type = meta["type"]
        self.instance = meta["instance"]
        self.n = self.body_id.size
        self.out_degree = np.diff(self.indptr)
        cs = np.concatenate(([0], np.cumsum(self.weight, dtype=np.int64)))
        self.out_syn = cs[self.indptr[1:]] - cs[self.indptr[:-1]]
        self.in_syn = np.bincount(self.indices, weights=self.weight, minlength=self.n).astype(np.int64)
        self.in_degree = np.bincount(self.indices, minlength=self.n)
        self._csc = None
        self._lock = threading.Lock()
        self._type_index = {}
        for i, t in enumerate(self.type):
            if t:
                self._type_index.setdefault(t.lower(), []).append(i)
        # Kenyon cell -> Kenyon cell synapses are mostly axo-axonal; in a point-neuron model they
        # make the mushroom body explode, so the simulation leaves them out (the data keeps them).
        is_kc = np.array([t.startswith("KC") for t in self.type])
        pre = np.repeat(is_kc, self.out_degree)
        self.sim_weight = np.where(pre & is_kc[self.indices], 0, self.weight).astype(np.float32)
        self.kc_kc_removed = int((pre & is_kc[self.indices]).sum())

    # ------------------------------------------------------------ lookups
    def label(self, kind, i):
        return self.vocab[kind][int(getattr(self, kind)[i])]

    def neurons_of_type(self, name):
        return self._type_index.get(name.lower(), [])

    def type_names(self):
        return self._type_index.keys()

    def mask(self, kind, values):
        codes = [k for k, v in enumerate(self.vocab[kind]) if v in values]
        return np.isin(getattr(self, kind), codes)

    def outputs(self, i):
        a, b = self.indptr[i], self.indptr[i + 1]
        return self.indices[a:b], self.weight[a:b]

    def inputs(self, i):
        if self._csc is None:
            order = np.argsort(self.indices, kind="stable")
            pre = np.repeat(np.arange(self.n, dtype=np.int32), self.out_degree)
            self._csc = (np.concatenate(([0], np.cumsum(self.in_degree))),
                         pre[order], self.weight[order])
        ptr, pre, w = self._csc
        return pre[ptr[i]:ptr[i + 1]], w[ptr[i]:ptr[i + 1]]

    def describe(self, i):
        i = int(i)
        return {
            "index": i, "bodyId": int(self.body_id[i]), "type": self.type[i],
            "instance": self.instance[i], "superclass": self.label("superclass", i),
            "class": self.label("cls", i), "subclass": self.label("subclass", i),
            "transmitter": self.label("nt", i), "side": self.label("side", i),
            "position_um": [round(float(v), 1) for v in self.pos[i]],
            "position_measured": bool(self.measured[i]),
            "inputs": int(self.in_degree[i]), "outputs": int(self.out_degree[i]),
            "input_synapses": int(self.in_syn[i]), "output_synapses": int(self.out_syn[i]),
        }

    def top_partners(self, i, direction="out", k=8):
        idx, w = self.outputs(i) if direction == "out" else self.inputs(i)
        order = np.argsort(w)[::-1][:k]
        return [{"index": int(idx[j]), "type": self.type[idx[j]] or "(untyped)",
                 "synapses": int(w[j])} for j in order]

    def sense_neurons(self, sense):
        spec = SENSES[sense]
        m = np.zeros(self.n, dtype=bool)
        if "cls" in spec:
            m |= self.mask("cls", spec["cls"])
        if "subclass" in spec:
            m |= self.mask("subclass", spec["subclass"])
        return np.flatnonzero(m)

    # ------------------------------------------------------------ simulation
    def new_state(self, seed=0):
        return SimState(self, seed)

    def simulate(self, stim, duration_ms=300.0, seed=0, frame_ms=10.0):
        """One-off experiment from a resting brain: drive `stim` neurons, return the response."""
        stim = np.asarray(stim, dtype=np.int64)
        with self._lock:
            st = SimState(self, seed)
            out = st.run(duration_ms, stim, frame_ms=frame_ms)
        out["stimulated"] = stim
        return out

    def summarize(self, result, top=12):
        counts, stim = result["counts"], result["stimulated"]
        rate = result["rate_hz"]
        is_stim = np.zeros(self.n, dtype=bool)
        is_stim[stim] = True
        responders = np.flatnonzero((counts > 0) & ~is_stim)
        by_sc = {}
        for code in np.unique(self.superclass[responders]):
            by_sc[self.vocab["superclass"][int(code)]] = int((self.superclass[responders] == code).sum())
        order = responders[np.argsort(rate[responders])[::-1]]
        dn_code = [k for k, v in enumerate(self.vocab["superclass"]) if v == "descending_neuron"]
        dns = order[np.isin(self.superclass[order], dn_code)]
        fmt = lambda i: {"index": int(i), "type": self.type[i] or "(untyped)",
                         "superclass": self.label("superclass", i), "rate_hz": round(float(rate[i]), 1)}
        return {"stimulated": int(stim.size), "responders": int(responders.size),
                "total_spikes": int(counts.sum()), "by_superclass": dict(sorted(by_sc.items(), key=lambda kv: -kv[1])),
                "top": [fmt(i) for i in order[:top]], "descending": [fmt(i) for i in dns[:top]]}


class SimState:
    """A brain whose membrane voltages, synaptic currents and adaptation persist between calls,
    so it can keep living in small chunks (FruitOS's always-on fly)."""

    def __init__(self, brain, seed=0):
        self.b = brain
        n = brain.n
        self.rng = np.random.default_rng(seed)
        self.dsteps = max(1, int(round(DELAY / DT)))
        self.v = np.full(n, V_REST, dtype=np.float32)
        self.g = np.zeros(n, dtype=np.float32)
        self.th = np.zeros(n, dtype=np.float32)
        self.refr = np.zeros(n, dtype=np.int32)
        self.pending = np.zeros((self.dsteps, n), dtype=np.float32)
        self.t = 0
        self.sw = brain.sign.astype(np.float32) * W_SYN

    def run(self, duration_ms, stim=(), stim_hz=STIM_HZ, noise=None, noise_hz=0.0, frame_ms=10.0):
        """Advance the brain. `stim` neurons get Poisson input at `stim_hz`; optional `noise`
        neurons get a background trickle at `noise_hz`."""
        b, n = self.b, self.b.n
        stim = np.asarray(stim, dtype=np.int64)
        noise = np.asarray(noise if noise is not None else [], dtype=np.int64)
        steps = int(duration_ms / DT)
        ref_steps = int(round(T_REF / DT))
        a_m = np.float32(DT / TAU_M)
        a_s = np.float32(np.exp(-DT / TAU_SYN))
        a_a = np.float32(np.exp(-DT / TAU_ADAPT))
        p_stim = stim_hz * DT / 1000.0
        p_noise = noise_hz * DT / 1000.0
        frame_steps = max(1, int(frame_ms / DT))
        counts = np.zeros(n, dtype=np.int32)
        frames, frame = [], []
        v, g, th, refr, pending = self.v, self.g, self.th, self.refr, self.pending
        for _ in range(steps):
            slot = self.t % self.dsteps
            self.t += 1
            g += pending[slot]
            pending[slot] = 0
            v += a_m * (V_REST - v + g)
            g *= a_s
            th *= a_a
            ext = stim[self.rng.random(stim.size) < p_stim] if stim.size else stim
            if noise.size:
                ext = np.concatenate((ext, noise[self.rng.random(noise.size) < p_noise]))
            active = refr <= 0
            spk = np.flatnonzero((v >= V_TH + th) & active)
            if ext.size:
                spk = np.union1d(spk, ext[active[ext]])
            refr -= 1
            if spk.size:
                v[spk] = V_RESET
                refr[spk] = ref_steps
                th[spk] += ADAPT_MV
                counts[spk] += 1
                starts = b.indptr[spk]
                lens = b.indptr[spk + 1] - starts
                total = int(lens.sum())
                if total:
                    # gather every outgoing synapse of every spiking neuron in one shot
                    offs = np.repeat(starts - np.concatenate(([0], np.cumsum(lens)[:-1])), lens)
                    edge = np.arange(total, dtype=np.int64) + offs
                    w = b.sim_weight[edge] * np.repeat(self.sw[spk], lens)
                    # this slot is read again exactly `dsteps` steps from now = the synaptic delay
                    pending[slot] += np.bincount(b.indices[edge], weights=w, minlength=n).astype(np.float32)
                frame.append(spk)
            if self.t % frame_steps == 0:
                frames.append(np.unique(np.concatenate(frame))[:MAX_FRAME_NEURONS].tolist() if frame else [])
                frame = []
        return {"counts": counts, "rate_hz": counts / (duration_ms / 1000.0), "frames": frames,
                "frame_ms": frame_ms, "duration_ms": duration_ms}

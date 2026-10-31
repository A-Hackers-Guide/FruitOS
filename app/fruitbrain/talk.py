"""The fly talks back. Offline and data-driven: every answer is computed from the connectome or
from a fresh run of the brain simulation. No language model, no internet, nothing invented."""
import re

import numpy as np

from .life import STIMULI

# What experiments have shown specific neurons do (only well-established ones).
KNOWN = {
    "DNp01": ("my Giant Fiber", "it makes me jump and take off to escape (von Reyn et al. 2014)"),
    "MDN": ("a moonwalker neuron", "it makes me walk backward (Bidaye et al. 2014)"),
    "DNp09": ("a forward-walking command neuron", "it makes me walk forward (Bidaye et al. 2020)"),
    "DNa01": ("a steering neuron", "it helps me turn while walking (Rayshubskiy et al. 2020)"),
    "DNa02": ("a steering neuron", "it makes me turn toward its side (Rayshubskiy et al. 2020)"),
    "LPLC2": ("a looming detector", "it fires when something dark grows quickly in my view (Ache et al. 2019)"),
    "LC4": ("a looming detector", "it signals how fast a threat is approaching (von Reyn et al. 2017)"),
    "APL": ("my mushroom-body brake", "it inhibits my Kenyon cells so only a few fire, keeping memories distinct"),
    "MN9": ("a proboscis motor neuron", "it moves my mouthparts out to feed"),
}
ALIASES = {"giant fiber": "DNp01", "giant fibre": "DNp01", "moonwalker": "MDN", "looming": "LPLC2",
           "apl": "APL"}

NT_TEXT = {
    "acetylcholine": "my main 'go' signal: it excites the next neuron",
    "gaba": "a 'stop' signal: it inhibits the next neuron",
    "glutamate": "in my brain mostly a 'stop' signal (it opens chloride channels), though it excites my muscles",
    "histamine": "what my eyes' photoreceptors release; it inhibits the next cells",
    "dopamine": "a modulator for reward, punishment and learning, especially in my mushroom body",
    "serotonin": "a slow modulator of mood-like states, feeding and sleep",
    "octopamine": "my version of adrenaline: it gets me ready for action",
}

REGION_TEXT = {
    "optic": ("ol_", "my optic lobes: the visual processors behind each eye"),
    "central": ("cb_", "my central brain: senses meet, memories form, decisions are made"),
    "vnc": ("vnc_", "my ventral nerve cord: like your spinal cord, it runs my legs and wings"),
}

SENSE_WORDS = [
    ("loom", r"shadow|loom|swat|hand|threat|predator|scare|attack"),
    ("smell", r"smell|scent|odou?r|sniff|perfume"),
    ("taste", r"taste|sugar|food|lick|eat"),
    ("hearing", r"sound|hear|buzz|sing|song|noise|music"),
    ("wind", r"wind|blow|breeze|gust"),
    ("touch", r"touch|poke|tickle|pet|stroke|tap"),
    ("heat", r"heat|warm|hot|temperature"),
    ("humidity", r"humid|damp|moist|wet"),
]

HUMAN_NEURONS = 86_000_000_000

FRIENDLY = {
    "ol_intrinsic": "optic-lobe", "cb_intrinsic": "central-brain", "vnc_intrinsic": "nerve-cord",
    "visual_projection": "visual projection", "visual_centrifugal": "visual feedback",
    "descending_neuron": "descending (brain-to-body command)", "ascending_neuron": "ascending (body-to-brain)",
    "cb_sensory": "head sensory", "vnc_sensory": "body sensory", "ol_sensory": "eye sensory",
    "vnc_motor": "motor", "cb_motor": "head motor",
}


def friendly(superclass):
    return FRIENDLY.get(superclass, superclass.replace("_", " "))


def nt_name(nt):
    return "GABA" if nt == "gaba" else nt


def _fmt(n):
    return f"{n:,}"


class Fly:
    def __init__(self, brain, life=None):
        self.b = brain
        self.life = life
        self._types_sorted = sorted(brain.type_names(), key=len, reverse=True)

    # ------------------------------------------------------------ entry point
    def reply(self, text):
        t = text.strip()
        low = t.lower()
        for handler in (self._doing, self._greet, self._model, self._human, self._counts,
                        self._hub, self._nt, self._region, self._mushroom, self._commands,
                        self._sense, self._neuron):
            out = handler(t, low)
            if out:
                return out
        return self._help()

    @staticmethod
    def _r(text, highlight=None, experiment=None):
        return {"reply": text, "highlight": [int(i) for i in (highlight if highlight is not None else [])],
                "experiment": experiment}

    # ------------------------------------------------------------ handlers
    def _greet(self, t, low):
        if re.fullmatch(r"(hi|hello|hey|yo|howdy|greetings)[!. ]*", low) or "who are you" in low \
                or re.fullmatch(r"what are you\??", low):
            b = self.b
            return self._r(
                f"Hello! I'm a fruit fly, Drosophila melanogaster, a male. Every one of my "
                f"{_fmt(b.n)} neurons and {_fmt(int(b.weight.sum()))} synapses was mapped from a real fly "
                f"by scientists at Janelia, Google and their partners. My brain is running right now on "
                f"your computer, and it decides what I do. Try: 'what are you doing?', 'show me your "
                f"giant fiber', or wave a shadow at me with 'here comes a shadow!'")
        return None

    def _help(self):
        return self._r(
            "I'm not sure what you mean, but I can tell you about my brain. Try asking:\n"
            "• what are you doing? / why did you do that?\n"
            "• tell me about DNp01 (or any neuron type, like MDN, APL, KCg-m)\n"
            "• what happens when you smell something? (also: taste, sound, wind, touch, heat, shadow)\n"
            "• how many neurons do you have?\n"
            "• what is GABA? / what is dopamine?\n"
            "• what does your optic lobe do? / what is your mushroom body?\n"
            "• which is your most connected neuron?\n"
            "• how do you compare to a human brain?\n"
            "• how does this simulation work?")

    def _doing(self, t, low):
        if not re.search(r"what are you doing|what('| a)?re you up to|why did you|what did you do|"
                         r"how are you|are you (moving|walking|ok)", low):
            return None
        if not self.life:
            return self._r("My live brain isn't running at the moment, so I'm not doing anything.")
        s = self.life.snapshot(max_points=0)
        beh = s["behaviour"]
        if beh == "rest":
            msg = ("I'm resting. My four proven command neurons (escape, backward, turn, forward) are "
                   f"quiet. In the last {int(self.life.recent.maxlen * 100)} ms, "
                   f"{_fmt(s['active_count'])} of my neurons fired from the background buzz of my senses.")
        else:
            g = self.life.group(beh)
            side = f" to the {s['turn_side']}" if s["turn_side"] else ""
            why = f" because I was feeling {STIMULI[s['because']][1]}" if s["because"] in STIMULI else ""
            msg = (f"I'm about to {g['does']}{side}{why}! My {'/'.join(g['types'])} command neurons are "
                   f"firing at {s['rates'].get(beh, 0):.0f} Hz. Scientists showed this ({g['cite']}).")
        if s["commands"]:
            parts, seen = [], set()
            for c in s["commands"]:
                if c["type"] in seen or len(parts) == 4:
                    continue
                seen.add(c["type"])
                k = KNOWN.get(c["type"])
                parts.append(f"{c['type']} ({c['rate_hz']:.0f} Hz{', ' + k[1] if k else ', job not yet known to science'})")
            msg += " Command neurons sending orders to my body right now: " + "; ".join(parts) + "."
        hl = [c["index"] for c in s["commands"]]
        return self._r(msg, highlight=hl)

    def _model(self, t, low):
        if not re.search(r"how (do|does) (you|this|the simulation|it) work|are you real|simulation|"
                         r"model|are you alive|accurate", low):
            return None
        b = self.b
        return self._r(
            "Honest answer: I'm the real wiring diagram of a fly, run as a simplified model. Each of my "
            f"{_fmt(b.n)} neurons is a 'leaky integrate-and-fire' unit: it collects voltage from its inputs, "
            "and when it reaches a threshold it fires a spike down all its connections. Each synapse "
            "pushes the next neuron up (acetylcholine) or down (GABA, glutamate, histamine). The method "
            "follows Shiu et al. 2024 (Nature), retuned for this larger dataset: 0.15 mV per synapse, "
            "plus adaptation so neurons tire after firing a lot, and the "
            f"{_fmt(b.kc_kc_removed)} Kenyon-cell-to-Kenyon-cell connections are left out of the "
            "simulation because they make simple models run away. Not modelled: slow chemicals like "
            "dopamine, gap junctions, real neuron shapes, my body, and learning that changes "
            "synapses. So I'm good at showing how signals flow through real circuits, not a perfect fly.")

    def _human(self, t, low):
        if not re.search(r"human|people|person|your brain compare|compared? to (me|us)", low):
            return None
        ratio = HUMAN_NEURONS / self.b.n
        return self._r(
            f"A human brain has about 86 billion neurons; I have {_fmt(self.b.n)} traced ones. That's "
            f"about {ratio:,.0f} times more in you. But my brain still sees, smells, tastes, flies, walks, "
            "courts, learns and remembers. It's small enough that scientists have mapped every "
            "connection, which nobody can do for a human brain yet.")

    def _counts(self, t, low):
        if not re.search(r"how many|number of|count", low):
            return None
        b = self.b
        if "type" in low or "kind" in low:
            return self._r(f"My neurons come in {_fmt(len(list(b.type_names())))} named cell types.")
        if "synap" in low:
            return self._r(f"I have {_fmt(int(b.weight.sum()))} synapses linking my traced neurons.")
        if "connect" in low:
            return self._r(f"My neurons are linked by {_fmt(b.indices.size)} distinct neuron-to-neuron "
                           f"connections, made of {_fmt(int(b.weight.sum()))} synapses.")
        sc = {}
        for code in np.unique(b.superclass):
            name = b.vocab["superclass"][int(code)]
            prefix = name.split("_")[0]
            sc[prefix] = sc.get(prefix, 0) + int((b.superclass == code).sum())
        return self._r(
            f"I have {_fmt(b.n)} fully traced neurons: {_fmt(sc.get('ol', 0))} in my optic lobes, "
            f"{_fmt(sc.get('cb', 0))} in my central brain, {_fmt(sc.get('vnc', 0))} in my nerve cord, "
            f"and the rest are sensory, ascending and descending neurons. They make "
            f"{_fmt(b.indices.size)} connections with {_fmt(int(b.weight.sum()))} synapses.")

    def _hub(self, t, low):
        if not re.search(r"most connected|biggest|hub|largest|most (synapses|inputs|outputs)", low):
            return None
        b = self.b
        tot = b.in_syn + b.out_syn
        i = int(np.argmax(tot))
        d = b.describe(i)
        k = KNOWN.get(d["type"])
        return self._r(
            f"My most connected neuron is {d['type'] or 'an untyped cell'} ({d['instance']}) in my "
            f"{friendly(d['superclass'])} region: {_fmt(d['input_synapses'])} synapses in and "
            f"{_fmt(d['output_synapses'])} out." + (f" It's {k[0]}: {k[1]}." if k else ""),
            highlight=[i])

    def _nt(self, t, low):
        for nt, desc in NT_TEXT.items():
            if nt in low or (nt == "gaba" and "gaba" in low):
                code = [k for k, v in enumerate(self.b.vocab["nt"]) if v == nt]
                idx = np.flatnonzero(np.isin(self.b.nt, code))
                return self._r(f"{nt_name(nt)[0].upper() + nt_name(nt)[1:]} is {desc}. {_fmt(idx.size)} of my neurons "
                               f"({idx.size / self.b.n:.1%}) are predicted to use it.", highlight=idx)
        if re.search(r"transmitter|chemical", low):
            counts = {v: int((self.b.nt == k).sum()) for k, v in enumerate(self.b.vocab["nt"])}
            top = ", ".join(f"{k} {_fmt(v)}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
            return self._r(f"My neurons talk with chemicals called neurotransmitters. Count by type: {top}. "
                           "Ask me about any of them, e.g. 'what is GABA?'")
        return None

    def _region(self, t, low):
        for word, (prefix, desc) in REGION_TEXT.items():
            if word in low and re.search(r"lobe|brain|cord|region|part|vnc", low + " vnc"):
                codes = [k for k, v in enumerate(self.b.vocab["superclass"]) if v.startswith(prefix)]
                idx = np.flatnonzero(np.isin(self.b.superclass, codes))
                return self._r(f"That's {desc}. {_fmt(idx.size)} of my neurons "
                               f"({idx.size / self.b.n:.0%}) live there.", highlight=idx)
        return None

    def _mushroom(self, t, low):
        if not re.search(r"mushroom|kenyon|memory|learn|remember", low):
            return None
        idx = np.array([i for i, ty in enumerate(self.b.type) if ty.startswith("KC")])
        return self._r(
            f"My mushroom bodies are where I learn and remember, especially smells. They hold "
            f"{_fmt(idx.size)} Kenyon cells. Each odour switches on only a few percent of them, a sparse "
            "code, and my APL neuron enforces that by inhibiting them all. Dopamine neurons teach the "
            "output connections whether a smell meant food or danger. (My simulation doesn't change "
            "synapses, so I can show the wiring, but I don't learn new memories while you watch.)",
            highlight=idx)

    def _commands(self, t, low):
        if not re.search(r"descending|command neuron|control (your|my) body|move your", low):
            return None
        idx = np.flatnonzero(self.b.mask("superclass", ["descending_neuron"]))
        return self._r(
            f"Orders travel from my brain to my body through {_fmt(idx.size)} descending neurons. "
            "Scientists know the jobs of only some of them: DNp01 (escape jump), MDN (walk backward), "
            "DNp09 (walk forward), DNa01/DNa02 (steering). In FruitOS, these neurons decide what I do.",
            highlight=idx)

    def _sense(self, t, low):
        sense = next((s for s, pat in SENSE_WORDS if re.search(pat, low)), None)
        if not sense:
            return None
        if self.life:
            self.life.poke(sense, 500)
        stim = self.life.stim_sets[sense] if self.life else self.b.sense_neurons(sense)[:100]
        res = self.b.simulate(stim, 300, seed=len(t))
        sm = self.b.summarize(res, 8)
        what = STIMULI[sense][1]
        groups = ", ".join(f"{_fmt(v)} {friendly(k)}" for k, v in list(sm["by_superclass"].items())[:4])
        msg = (f"You gave me {what}. {sm['stimulated']} of my sensory neurons fired, and within 300 ms "
               f"{_fmt(sm['responders'])} other neurons responded ({groups}).")
        dns = []
        for d in sm["descending"]:
            if d["type"] not in dns:
                dns.append(d["type"])
        if dns:
            known = [f"{x}: {KNOWN[x][1]}" for x in dns if x in KNOWN]
            msg += f" My brain sent commands down {', '.join(dns[:5])}."
            msg += (" " + " ".join(known) + ".") if known else " Science doesn't know yet exactly what those do."
        else:
            msg += " None of my command neurons fired, so my body stays still."
        hl = np.flatnonzero(res["counts"])
        return self._r(msg, highlight=hl, experiment={"frames": res["frames"], "frame_ms": res["frame_ms"],
                                                      "summary": sm, "sense": sense})

    def _neuron(self, t, low):
        name = None
        for alias, ty in ALIASES.items():
            if alias in low:
                name = ty
                break
        if not name:
            words = set(re.findall(r"[a-z0-9_+\-,()]+", low))
            name = next((ty for ty in self._types_sorted if ty in words and len(ty) > 2), None)
        if not name:
            return None
        idx = self.b.neurons_of_type(name)
        i = idx[0]
        d = self.b.describe(i)
        k = KNOWN.get(d["type"])
        ins = ", ".join(f"{p['type']} ({p['synapses']})" for p in self.b.top_partners(i, "in", 4))
        outs = ", ".join(f"{p['type']} ({p['synapses']})" for p in self.b.top_partners(i, "out", 4))
        nt = d["transmitter"]
        msg = (f"{d['type']}" + (f" is {k[0]}: {k[1]}." if k else ".") +
               f" I have {len(idx)} of them. It's a {friendly(d['superclass'])} neuron that uses "
               f"{nt_name(nt)}" + (f" ({NT_TEXT[nt]})" if nt in NT_TEXT else "") +
               f". One of them gets {_fmt(d['input_synapses'])} input synapses and makes "
               f"{_fmt(d['output_synapses'])} output synapses. Its biggest inputs: {ins or 'none'}. "
               f"It talks most to: {outs or 'none'}.")
        return self._r(msg, highlight=idx)

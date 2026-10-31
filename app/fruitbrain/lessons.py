"""Guided lessons. Each lesson's numbers and quiz answers are computed from the connectome, so
they are always true for the brain the learner is looking at."""
import random

import numpy as np


def _pct(x):
    return f"{x:.0%}"


def build(brain):
    b = brain
    sc = b.vocab["superclass"]
    in_region = lambda prefix: int(np.isin(b.superclass, [k for k, v in enumerate(sc) if v.startswith(prefix)]).sum())
    sensory = int(np.isin(b.superclass, [k for k, v in enumerate(sc) if "sensory" in v]).sum())
    dns = int(b.mask("superclass", ["descending_neuron"]).sum())
    kcs = sum(1 for t in b.type if t.startswith("KC"))
    nt_counts = {v: int((b.nt == k).sum()) for k, v in enumerate(b.vocab["nt"])}
    top_nt = max(nt_counts, key=nt_counts.get)
    cls_counts = {}
    for k, v in enumerate(b.vocab["cls"]):
        if v in ("visual", "olfactory", "gustatory", "mechanosensory_tactile", "thermosensory", "hygrosensory"):
            cls_counts[v] = int((b.cls == k).sum())
    top_sense = max(cls_counts, key=cls_counts.get)
    tot = b.in_syn + b.out_syn
    hub = int(np.argmax(tot))
    gf = b.neurons_of_type("DNp01")
    gf_nt = b.label("nt", gf[0]) if gf else "acetylcholine"
    ol = in_region("ol_")

    def q(question, answer, wrong):
        choices = [answer] + [w for w in wrong if w != answer][:3]
        random.Random(question).shuffle(choices)
        return {"question": question, "choices": choices, "answer": answer}

    def num(n):
        return f"{n:,}"

    return [
        {"id": "meet", "title": "1. Meet my brain",
         "text": (f"I'm a male fruit fly. Scientists sliced my nervous system into thousands of ultra-thin "
                  f"layers, photographed them with an electron microscope, and traced every neuron. You're "
                  f"looking at {num(b.n)} of them, each dot placed where its cell body sits. Drag to spin "
                  f"me, scroll to zoom."),
         "action": {"highlight": "all"},
         "quiz": q("How many fully traced neurons does this fly have?", num(b.n),
                   [num(b.n // 10), num(b.n * 10), "86,000,000,000"])},
        {"id": "wires", "title": "2. Neurons and synapses",
         "text": (f"A neuron passes signals to others at junctions called synapses. My neurons make "
                  f"{num(b.indices.size)} neuron-to-neuron connections from {num(int(b.weight.sum()))} "
                  f"synapses. Some neurons are giant hubs: the glowing one is my most connected neuron, "
                  f"{b.type[hub] or 'an untyped cell'}."),
         "action": {"highlight": [hub]},
         "quiz": q("What is a synapse?", "A junction where one neuron passes a signal to another",
                   ["A type of brain region", "The cell body of a neuron", "A sense organ"])},
        {"id": "chemicals", "title": "3. Go and stop signals",
         "text": (f"Neurons talk with chemicals. Acetylcholine says 'go' (excites); GABA and glutamate "
                  f"mostly say 'stop' (inhibit). The balance between go and stop is what keeps a brain "
                  f"working instead of exploding with activity. My most common transmitter is {top_nt}: "
                  f"{num(nt_counts[top_nt])} neurons."),
         "action": {"highlight_nt": top_nt},
         "quiz": q("Which transmitter do most of this fly's neurons use?", top_nt,
                   ["gaba", "dopamine", "serotonin", "glutamate"])},
        {"id": "senses", "title": "4. My senses",
         "text": (f"{num(sensory)} of my neurons are sensory: they turn light, smells, tastes, sounds, "
                  f"touch, heat and humidity into spikes. Press 'Smell' below and watch the signal enter "
                  f"through my antennae and spread into my brain."),
         "action": {"poke": "smell"},
         "quiz": q("Which sense has the most sensory neurons in this fly?", top_sense.replace("_", " "),
                   ["olfactory", "gustatory", "thermosensory", "visual"])},
        {"id": "vision", "title": "5. A brain built for seeing",
         "text": (f"{_pct(ol / b.n)} of my neurons ({num(ol)}) are in my optic lobes, the visual processors "
                  f"behind my eyes. Flying at high speed means processing images fast: that's why so much "
                  f"of my brain is about seeing."),
         "action": {"highlight_prefix": "ol_"},
         "quiz": q("Roughly what share of this fly's neurons are in the optic lobes?", _pct(ol / b.n),
                   ["5%", "25%", "90%"])},
        {"id": "escape", "title": "6. The escape circuit (I'm in control!)",
         "text": ("When something dark rushes toward me, looming-detector neurons (LPLC2, LC4) fire. They "
                  "drive my Giant Fiber, DNp01, which makes me jump and take off in milliseconds. Press "
                  "'Shadow' and watch: nobody scripted my reaction. My own wiring decides it."),
         "action": {"poke": "loom", "highlight_types": ["LPLC2", "LC4", "DNp01"]},
         "quiz": q("Which neuron makes the fly jump to escape?", "DNp01 (the Giant Fiber)",
                   ["MDN (moonwalker)", "APL", "A Kenyon cell"])},
        {"id": "memory", "title": "7. Learning and memory",
         "text": (f"My mushroom bodies store memories, especially of smells. {num(kcs)} Kenyon cells each "
                  f"respond to particular combinations of smells, and dopamine neurons label those "
                  f"memories as good or bad."),
         "action": {"highlight_kc": True},
         "quiz": q("Which brain structure does a fly use to learn about smells?", "The mushroom body",
                   ["The optic lobe", "The ventral nerve cord", "The Giant Fiber"])},
        {"id": "body", "title": "8. From brain to body",
         "text": (f"{num(dns)} descending neurons carry commands from my brain down to my ventral nerve "
                  f"cord, which runs my legs and wings, like your spinal cord. In FruitOS, the firing of "
                  f"these command neurons, and nothing else, decides what I do."),
         "action": {"highlight_superclass": "descending_neuron"},
         "quiz": q("What do descending neurons do?", "Carry commands from the brain to the body",
                   ["Detect smells", "Store memories", "Process images"])},
        {"id": "limits", "title": "9. What a model can and can't tell you",
         "text": ("I'm the real wiring of a real fly, but run as a simplified model: neurons are simple "
                  "voltage counters, slow chemicals like dopamine aren't simulated, and my synapses don't "
                  "learn while you watch. Good science says what a model leaves out. Ask me 'how does "
                  "this simulation work?' for the details."),
         "action": {},
         "quiz": q("Which of these is NOT simulated in FruitOS?", "Slow chemicals like dopamine changing synapses",
                   ["Spikes travelling along real connections", "Inhibition by GABA",
                    "The Giant Fiber escape pathway"])},
    ]

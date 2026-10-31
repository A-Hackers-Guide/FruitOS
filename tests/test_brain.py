"""Engine tests on a tiny hand-made brain, so CI needs no 1 GB download."""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from fruitbrain.brain import Brain  # noqa: E402
from fruitbrain.talk import Fly  # noqa: E402


def tiny_brain(d):
    # 0 sensory -> 1 relay (strong) -> 2 command ; 3 is an inhibitory cell onto 2 ; 4 is isolated
    pre = np.array([0, 1, 3]); post = np.array([1, 2, 2]); w = np.array([200, 200, 400])
    order = np.argsort(pre, kind="stable")
    indptr = np.concatenate(([0], np.cumsum(np.bincount(pre, minlength=5))))
    vocab = {"superclass": ["cb_intrinsic", "cb_sensory", "descending_neuron"], "cls": ["gustatory", "unknown"],
             "subclass": ["unknown"], "nt": ["acetylcholine", "gaba"], "side": ["L", "R"]}
    np.savez(os.path.join(d, "brain.npz"), indptr=indptr, indices=post[order].astype(np.int32),
             weight=w[order].astype(np.uint16), sign=np.array([1, 1, 1, -1, 1], np.int8),
             pos=np.zeros((5, 3), np.float32), measured=np.ones(5, bool),
             superclass=np.array([1, 0, 2, 0, 0], np.uint16), cls=np.array([0, 1, 1, 1, 1], np.uint16),
             subclass=np.zeros(5, np.uint16), nt=np.array([0, 0, 0, 1, 0], np.uint16),
             side=np.zeros(5, np.uint16))
    json.dump({"vocab": vocab, "body_id": [10, 11, 12, 13, 14],
               "type": ["GRN_x", "relay", "DNp01", "inhib", "lonely"], "instance": [""] * 5},
              open(os.path.join(d, "meta.json"), "w"))


class BrainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        tiny_brain(cls.tmp.name)
        cls.b = Brain(cls.tmp.name)

    def test_quiet_without_input(self):
        self.assertEqual(int(self.b.simulate([], 50)["counts"].sum()), 0)

    def test_signal_reaches_command_neuron(self):
        c = self.b.simulate([0], 200)["counts"]
        self.assertGreater(c[1], 0)
        self.assertGreater(c[2], 0)
        self.assertEqual(c[4], 0)

    def test_inhibition_suppresses(self):
        alone = self.b.simulate([0], 200, seed=1)["counts"][2]
        with_inh = self.b.simulate([0, 3], 200, seed=1)["counts"][2]
        self.assertLess(with_inh, alone)

    def test_partners(self):
        self.assertEqual(self.b.top_partners(2, "in")[0]["type"], "inhib")
        self.assertEqual(self.b.top_partners(0, "out")[0]["type"], "relay")

    def test_fly_answers_from_data(self):
        f = Fly(self.b)
        self.assertIn("5 fully traced neurons", f.reply("how many neurons do you have?")["reply"])
        self.assertIn("Giant Fiber", f.reply("tell me about DNp01")["reply"])


if __name__ == "__main__":
    unittest.main()

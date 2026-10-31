"""FruitOS local server: serves the 3D learning app and the fly's brain on 127.0.0.1 only.

    python3 -m fruitbrain.server            # http://127.0.0.1:8642
Environment: FRUITOS_DATA (brain files), FRUITOS_PORT, FRUITOS_CPU_SHARE (0-1 of one core).
"""
import json
import mimetypes
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

from . import __version__, lessons
from .brain import Brain
from .life import STIMULI, Life
from .talk import Fly

WEB = os.environ.get("FRUITOS_WEB", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web"))
PORT = int(os.environ.get("FRUITOS_PORT", "8642"))


class App:
    def __init__(self):
        print("FruitOS: loading the fly's brain...", flush=True)
        self.brain = Brain()
        self.life = Life(self.brain)
        self.fly = Fly(self.brain, self.life)
        self.lessons = lessons.build(self.brain)
        b = self.brain
        p = b.pos - b.pos.mean(axis=0)
        p /= np.abs(p).max()
        self.points = (p.astype(np.float32).tobytes() + b.superclass.astype(np.uint8).tobytes()
                       + b.measured.astype(np.uint8).tobytes())
        threading.Thread(target=self.brain.inputs, args=(0,), daemon=True).start()  # build reverse index
        self.life.start()
        print(f"FruitOS: {b.n:,} neurons alive at http://127.0.0.1:{PORT}", flush=True)

    def info(self):
        b = self.brain
        return {"version": __version__, "neurons": b.n, "connections": int(b.indices.size),
                "synapses": int(b.weight.sum()), "superclass": b.vocab["superclass"],
                "senses": {k: v[1] for k, v in STIMULI.items()},
                "lessons": [{"id": l["id"], "title": l["title"]} for l in self.lessons],
                "data_credit": "MaleCNS v1.0: Janelia FlyEM, Google, MRC LMB, Cambridge and partners (CC BY 4.0)"}

    def lesson(self, lid):
        l = next((x for x in self.lessons if x["id"] == lid), None)
        if not l:
            return None
        b, a = self.brain, l["action"]
        hl = []
        if a.get("highlight") == "all":
            hl = "all"
        elif isinstance(a.get("highlight"), list):
            hl = a["highlight"]
        elif "highlight_nt" in a:
            hl = np.flatnonzero(b.mask("nt", [a["highlight_nt"]])).tolist()
        elif "highlight_prefix" in a:
            codes = [k for k, v in enumerate(b.vocab["superclass"]) if v.startswith(a["highlight_prefix"])]
            hl = np.flatnonzero(np.isin(b.superclass, codes)).tolist()
        elif "highlight_superclass" in a:
            hl = np.flatnonzero(b.mask("superclass", [a["highlight_superclass"]])).tolist()
        elif a.get("highlight_kc"):
            hl = [i for i, t in enumerate(b.type) if t.startswith("KC")]
        if "highlight_types" in a:
            hl = [i for t in a["highlight_types"] for i in b.neurons_of_type(t)]
        if "poke" in a and not a.get("highlight_types"):
            hl = self.life.stim_sets[a["poke"]].tolist()
        return {**l, "highlight": hl}

    def search(self, q):
        q = q.lower().strip()
        if not q:
            return []
        b = self.brain
        hits = [t for t in b.type_names() if q in t]
        hits.sort(key=lambda t: (not t.startswith(q), len(t)))
        out = []
        for t in hits[:20]:
            idx = b.neurons_of_type(t)
            out.append({"type": b.type[idx[0]], "count": len(idx), "indices": idx[:500],
                        "superclass": b.label("superclass", idx[0])})
        return out

    def neuron(self, i):
        b = self.brain
        return {**b.describe(i), "top_inputs": b.top_partners(i, "in", 8),
                "top_outputs": b.top_partners(i, "out", 8)}


APP = None


class Handler(BaseHTTPRequestHandler):
    server_version = "FruitOS"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        if not isinstance(body, (bytes, bytearray)):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/api/info":
                return self._send(200, APP.info())
            if u.path == "/api/points":
                return self._send(200, APP.points, "application/octet-stream")
            if u.path == "/api/live":
                APP.life.heartbeat()
                return self._send(200, APP.life.snapshot())
            if u.path == "/api/neuron":
                return self._send(200, APP.neuron(int(q["i"])))
            if u.path == "/api/search":
                return self._send(200, APP.search(q.get("q", "")))
            if u.path == "/api/lesson":
                l = APP.lesson(q.get("id", ""))
                return self._send(200 if l else 404, l or {"error": "no such lesson"})
            return self._static(u.path)
        except (KeyError, ValueError, IndexError) as e:
            return self._send(400, {"error": str(e)})

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length", "0") or 0)
        if n > 100_000:
            return self._send(413, {"error": "too large"})
        try:
            data = json.loads(self.rfile.read(n) or b"{}")
            if u.path == "/api/chat":
                return self._send(200, APP.fly.reply(str(data.get("text", ""))[:500]))
            if u.path == "/api/poke":
                sense = data.get("sense")
                if sense not in STIMULI:
                    return self._send(400, {"error": "unknown sense"})
                APP.life.poke(sense, 500)
                return self._send(200, {"ok": True})
        except (ValueError, KeyError) as e:
            return self._send(400, {"error": str(e)})
        return self._send(404, {"error": "not found"})

    def _static(self, path):
        rel = "index.html" if path in ("", "/") else path.lstrip("/")
        full = os.path.realpath(os.path.join(WEB, rel))
        if not full.startswith(os.path.realpath(WEB) + os.sep) or not os.path.isfile(full):
            return self._send(404, {"error": "not found"})
        with open(full, "rb") as f:
            body = f.read()
        ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
        if full.endswith(".js"):
            ctype = "text/javascript"
        return self._send(200, body, ctype)


def main():
    global APP
    APP = App()
    httpd = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()

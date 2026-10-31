"""Turn the MaleCNS v1.0 flat-connectome release into the compact files FruitOS ships.

Build-time only (needs pyarrow + pandas). The OS itself only needs numpy + scipy.

    python3 tools/prepare_data.py --raw RAW_DIR --out OUT_DIR

RAW_DIR must hold the three files listed in RAW_FILES (download with tools/fetch_data.sh).
Data: Janelia FlyEM / Google MaleCNS v1.0, CC BY 4.0. See DATA_LICENSE.md.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import pyarrow.feather as pf
import pyarrow.ipc as ipc
from scipy import sparse

RAW_FILES = {
    "ann": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "nt": "body-neurotransmitters-male-cns-v1.0.feather",
    "w": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
}
VOXEL_UM = 0.008  # 8 nm voxels -> micrometres

# Fast sign of each transmitter in the simulation. Modulators act slowly and are not modelled (0).
NT_SIGN = {"acetylcholine": 1, "gaba": -1, "glutamate": -1, "histamine": -1,
           "dopamine": 0, "serotonin": 0, "octopamine": 0, "unclear": 1, "unknown": 1}


def codes(series):
    s = series.fillna("unknown").astype(str)
    vocab = sorted(s.unique())
    lut = {v: i for i, v in enumerate(vocab)}
    return np.array([lut[v] for v in s], dtype=np.uint16), vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    raw = {k: os.path.join(args.raw, v) for k, v in RAW_FILES.items()}

    # ---- neurons: every fully traced body
    ann = pf.read_table(raw["ann"]).to_pandas()
    ann = ann[ann.status == "Traced"].sort_values("bodyId").reset_index(drop=True)
    ids = ann.bodyId.to_numpy(np.int64)
    n = ids.size
    print(f"neurons: {n:,}")

    nt = pf.read_table(raw["nt"], columns=["body", "consensus_nt", "predicted_nt"]).to_pandas()
    nt = nt.drop_duplicates("body").set_index("body")
    nt_name = nt.consensus_nt.where(nt.consensus_nt.notna() & (nt.consensus_nt != "unclear"), nt.predicted_nt)
    ann["nt"] = ann.bodyId.map(nt_name).fillna("unknown")

    # ---- edges among traced neurons, streamed in batches to keep RAM low
    pres, posts, ws = [], [], []
    reader = ipc.open_file(raw["w"])
    for i in range(reader.num_record_batches):
        b = reader.get_batch(i)
        pre = b.column("body_pre").to_numpy()
        post = b.column("body_post").to_numpy()
        w = b.column("weight").to_numpy()
        pi = np.searchsorted(ids, pre).clip(0, n - 1)
        qi = np.searchsorted(ids, post).clip(0, n - 1)
        keep = (ids[pi] == pre) & (ids[qi] == post) & (pi != qi)
        pres.append(pi[keep].astype(np.int32))
        posts.append(qi[keep].astype(np.int32))
        ws.append(np.minimum(w[keep], 65535).astype(np.uint16))
    pre = np.concatenate(pres); post = np.concatenate(posts); w = np.concatenate(ws)
    del pres, posts, ws
    W = sparse.csr_matrix((w.astype(np.float32), (pre, post)), shape=(n, n))
    W.sum_duplicates()
    del pre, post, w
    print(f"connections: {W.nnz:,}  synapses: {int(W.data.sum()):,}")

    # ---- 3D positions: soma where measured, else the synapse-weighted centre of its partners
    pos = np.full((n, 3), np.nan, dtype=np.float64)
    has = ann.somaLocation.notna().to_numpy()
    pos[has] = np.stack(ann.somaLocation[has].to_numpy()) * VOXEL_UM
    measured = has.copy()
    A = (W + W.T).tocsr()
    for _ in range(4):
        known = ~np.isnan(pos[:, 0])
        miss = np.flatnonzero(~known)
        if miss.size == 0:
            break
        Am = A[miss][:, np.flatnonzero(known)]
        tot = np.asarray(Am.sum(axis=1)).ravel()
        ok = tot > 0
        est = (Am @ pos[known]) / np.maximum(tot, 1)[:, None]
        pos[miss[ok]] = est[ok]
    left = np.isnan(pos[:, 0])
    pos[left] = np.nanmean(pos, axis=0)
    rng = np.random.default_rng(0)
    pos[~measured] += rng.normal(0, 4.0, size=(int((~measured).sum()), 3))  # spread stacked estimates
    print(f"positions: {int(measured.sum()):,} measured, {int((~measured).sum()):,} estimated")

    sc, sc_vocab = codes(ann.superclass)
    cl, cl_vocab = codes(ann["class"])
    sub, sub_vocab = codes(ann.subclass)
    ntc, nt_vocab = codes(ann.nt)
    side, side_vocab = codes(ann.somaSide.fillna(ann.rootSide))
    sign = np.array([NT_SIGN.get(v, 1) for v in ann.nt], dtype=np.int8)

    np.savez(os.path.join(args.out, "brain.npz"),
             indptr=W.indptr.astype(np.int64), indices=W.indices.astype(np.int32),
             weight=W.data.astype(np.uint16), sign=sign, pos=pos.astype(np.float32),
             measured=measured, superclass=sc, cls=cl, subclass=sub, nt=ntc, side=side)
    meta = {
        "source": "MaleCNS v1.0 (Janelia FlyEM, Google, et al.), CC BY 4.0",
        "vocab": {"superclass": sc_vocab, "cls": cl_vocab, "subclass": sub_vocab,
                  "nt": nt_vocab, "side": side_vocab},
        "body_id": ids.tolist(),
        "type": ann.type.fillna("").astype(str).tolist(),
        "instance": ann.instance.fillna("").astype(str).tolist(),
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, separators=(",", ":"))
    for fn in ("brain.npz", "meta.json"):
        print(f"{fn}: {os.path.getsize(os.path.join(args.out, fn)) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()

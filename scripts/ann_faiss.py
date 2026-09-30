"""The same SIFT1M benchmark as `ferrolm ann`, run with FAISS, for a
like-for-like comparison: identical data, parameters, query set, recall
definition (10@10) and thread counts.

Usage: python scripts/ann_faiss.py data/sift hnsw|ivfpq OUT.jsonl
"""

import json
import os
import sys
import time

import faiss
import numpy as np


def fvecs(path):
    a = np.fromfile(path, dtype="int32")
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy().view("float32")


def ivecs(path):
    a = np.fromfile(path, dtype="int32")
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:]


def recall(found, truth, k=10):
    return float(np.mean([len(set(f[:k]) & set(t[:k])) / k for f, t in zip(found, truth)]))


def main(root, kind, out):
    xb = fvecs(os.path.join(root, "sift_base.fvecs"))
    xq = fvecs(os.path.join(root, "sift_query.fvecs"))
    gt = ivecs(os.path.join(root, "sift_groundtruth.ivecs"))
    d = xb.shape[1]
    threads = os.cpu_count()
    sub = 2000

    def run(index, label, param, build):
        faiss.omp_set_num_threads(threads)
        t = time.time()
        _, found = index.search(xq, 10)
        qpsn = len(xq) / (time.time() - t)
        faiss.omp_set_num_threads(1)
        t = time.time()
        index.search(xq[:sub], 10)
        qps1 = sub / (time.time() - t)
        r = recall(found, gt)
        print(f"faiss {label} {param}: recall@10 {r:.4f}  {qps1:.0f} QPS (1 thread)  {qpsn:.0f} QPS ({threads} threads)  build {build:.0f} s")
        with open(out, "a") as f:
            f.write(json.dumps({"engine": "faiss", "index": label, "param": param, "recall10": r, "qps_1": qps1,
                                "qps_n": qpsn, "threads": threads, "build_s": build, "n": len(xb)}) + "\n")

    faiss.omp_set_num_threads(threads)
    if kind == "hnsw":
        index = faiss.IndexHNSWFlat(d, 16)
        index.hnsw.efConstruction = 200
        t = time.time()
        index.add(xb)
        build = time.time() - t
        for ef in [16, 32, 64, 128, 256]:
            index.hnsw.efSearch = ef
            run(index, "hnsw", f"M=16 efC=200 ef={ef}", build)
    else:
        xt = fvecs(os.path.join(root, "sift_learn.fvecs"))
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFPQ(quantizer, d, 1024, 16, 8)
        t = time.time()
        index.train(xt)
        index.add(xb)
        build = time.time() - t
        for refine in [0, 10]:
            ix = index
            if refine:
                ix = faiss.IndexRefineFlat(index, faiss.swig_ptr(xb))
                ix.k_factor = refine
            for nprobe in [1, 4, 16, 64]:
                index.nprobe = nprobe
                run(ix, "ivfpq", f"nlist=1024 m=16 nprobe={nprobe} refine={refine}", build)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])

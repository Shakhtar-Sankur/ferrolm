"""Draws the README's charts (SVG, no dependencies) from bench/results/*.jsonl.

Usage: python3 scripts/charts.py
"""

import json
import math
import os

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834"]  # validated categorical slots 1-2
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
W = 720


def load(name):
    path = os.path.join("bench/results", name)
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def header(h, title, subtitle):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{h}" viewBox="0 0 {W} {h}" '
        f'font-family="{FONT}" role="img" aria-label="{esc(title)}">',
        f'<rect width="{W}" height="{h}" rx="8" fill="{SURFACE}"/>',
        f'<text x="24" y="34" font-size="17" font-weight="600" fill="{TEXT}">{esc(title)}</text>',
        f'<text x="24" y="56" font-size="13" fill="{TEXT2}">{esc(subtitle)}</text>',
    ]


def nice_max(v):
    if v <= 0:
        return 1
    e = 10 ** math.floor(math.log10(v))
    for m in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if m * e >= v:
            return m * e
    return 10 * e


def nice_ticks(v):
    """Round tick values from 0 covering v, at most 6 intervals."""
    e = 10 ** math.floor(math.log10(v))
    for m in (0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10):
        step = m * e
        if v / step <= 6:
            n = math.ceil(v / step - 1e-9)
            return [step * i for i in range(n + 1)]
    return [0, v]


def bars(path, title, subtitle, rows, unit, fmt="{:.0f}", note=None):
    """Horizontal bars: rows of (label, value, emphasised)."""
    left, right, top, band = 170, 110, 80, 40
    h = top + band * len(rows) + (60 if note else 36)
    ticks = nice_ticks(max(v for _, v, _ in rows))
    vmax = ticks[-1]
    plot = W - left - right
    out = header(h, title, subtitle)
    for t in ticks:
        x = left + plot * t / vmax
        out.append(f'<line x1="{x:.1f}" y1="{top - 6}" x2="{x:.1f}" y2="{top + band * len(rows)}" stroke="{GRID}" stroke-width="1"/>')
        out.append(f'<text x="{x:.1f}" y="{top + band * len(rows) + 18}" font-size="12" fill="{TEXT2}" text-anchor="middle">{t:g}</text>')
    for i, (label, v, strong) in enumerate(rows):
        y = top + band * i + (band - 24) / 2
        w = max(plot * v / vmax, 4)
        color = SERIES[0] if strong else "#9ec5f4"
        # 4px rounded data end, square at the baseline.
        out.append(
            f'<path d="M{left},{y} h{w - 4:.1f} a4,4 0 0 1 4,4 v16 a4,4 0 0 1 -4,4 h{-(w - 4):.1f} z" fill="{color}">'
            f"<title>{esc(label)}: {fmt.format(v)} {esc(unit)}</title></path>"
        )
        out.append(f'<text x="{left - 12}" y="{y + 16}" font-size="13" fill="{TEXT}" text-anchor="end">{esc(label)}</text>')
        out.append(f'<text x="{left + w + 8:.1f}" y="{y + 16}" font-size="13" font-weight="600" fill="{TEXT}">{fmt.format(v)} <tspan font-weight="400" fill="{TEXT2}">{esc(unit)}</tspan></text>')
    if note:
        out.append(f'<text x="24" y="{h - 16}" font-size="12" fill="{TEXT2}">{esc(note)}</text>')
    out.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(out))


def lines(path, title, subtitle, xs, series, xlabel, ylabel, yfmt="{:.0f}", log=False, unit=""):
    """Lines with end labels and a legend: series is [(name, ys)]."""
    left, right, top, bottom = 64, 70, 96, 56
    h = 400
    plot_w, plot_h = W - left - right, h - top - bottom
    ys_all = [y for _, ys in series for y in ys if y > 0]
    if log:
        lo = 10 ** math.floor(math.log10(min(ys_all)))
        lo = 5 * lo if min(ys_all) >= 5 * lo else lo
        hi = 10 ** math.ceil(math.log10(max(ys_all)))
        ticks = [lo] if math.log10(lo) % 1 else []
        t = 10 ** math.ceil(math.log10(lo))
        while t <= hi * 1.001:
            ticks.append(t)
            t *= 10
        ty = lambda v: top + plot_h * (1 - (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)))
    else:
        hi = nice_max(max(ys_all))
        ticks = [hi * i / 4 for i in range(5)]
        ty = lambda v: top + plot_h * (1 - v / hi)
    x0, x1 = min(xs), max(xs)
    tx = lambda v: left + plot_w * (v - x0) / (x1 - x0)
    out = header(h, title, subtitle)
    # Legend, one row under the subtitle.
    lx = 24
    for i, (name, _) in enumerate(series):
        out.append(f'<line x1="{lx}" y1="74" x2="{lx + 18}" y2="74" stroke="{SERIES[i]}" stroke-width="2" stroke-linecap="round"/>')
        out.append(f'<circle cx="{lx + 9}" cy="74" r="4" fill="{SERIES[i]}" stroke="{SURFACE}" stroke-width="2"/>')
        out.append(f'<text x="{lx + 26}" y="78" font-size="13" fill="{TEXT}">{esc(name)}</text>')
        lx += 40 + 7.5 * len(name)
    for t in ticks:
        y = ty(t)
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        out.append(f'<text x="{left - 8}" y="{y + 4:.1f}" font-size="12" fill="{TEXT2}" text-anchor="end">{yfmt.format(t)}</text>')
    for x in xs:
        out.append(f'<text x="{tx(x):.1f}" y="{top + plot_h + 20}" font-size="12" fill="{TEXT2}" text-anchor="middle">{x:g}</text>')
    out.append(f'<text x="{left + plot_w / 2}" y="{h - 14}" font-size="12" fill="{TEXT2}" text-anchor="middle">{esc(xlabel)}</text>')
    out.append(f'<text x="16" y="{top + plot_h / 2}" font-size="12" fill="{TEXT2}" text-anchor="middle" transform="rotate(-90 16 {top + plot_h / 2})">{esc(ylabel)}</text>')
    ends = []
    for i, (name, ys) in enumerate(series):
        pts = " ".join(f"{tx(x):.1f},{ty(max(y, 1e-9)):.1f}" for x, y in zip(xs, ys))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{SERIES[i]}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        for x, y in zip(xs, ys):
            out.append(f'<circle cx="{tx(x):.1f}" cy="{ty(max(y, 1e-9)):.1f}" r="4" fill="{SERIES[i]}" stroke="{SURFACE}" stroke-width="2"><title>{esc(name)} at {x:g}: {yfmt.format(y)} {esc(ylabel)}</title></circle>')
        ends.append([ty(ys[-1]), name, ys[-1]])
    # End labels, kept apart.
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 18)
    for y, name, v in ends:
        out.append(f'<text x="{left + plot_w + 10}" y="{y + 4:.1f}" font-size="13" font-weight="600" fill="{TEXT}">{yfmt.format(v)}<tspan font-weight="400" fill="{TEXT2}"> {esc(unit)}</tspan></text>')
    out.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(out))


def tradeoff(path, title, subtitle, panels, note):
    """Small multiples of recall (x) against queries/s (log y): panels is
    [(panel title, [(series name, [(recall, qps, param)])])], same series
    order in each so colour follows the engine."""
    top, bottom, gap, left = 124, 62, 56, 84
    h = 440
    pw = (W - left - 24 - gap * (len(panels) - 1)) / len(panels)
    ph = h - top - bottom
    out = header(h, title, subtitle)
    lx = 24
    for i, (name, _) in enumerate(panels[0][1]):
        out.append(f'<line x1="{lx}" y1="74" x2="{lx + 18}" y2="74" stroke="{SERIES[i]}" stroke-width="2" stroke-linecap="round"/>')
        out.append(f'<circle cx="{lx + 9}" cy="74" r="4" fill="{SERIES[i]}" stroke="{SURFACE}" stroke-width="2"/>')
        out.append(f'<text x="{lx + 26}" y="78" font-size="13" fill="{TEXT}">{esc(name)}</text>')
        lx += 40 + 7.5 * len(name)
    for k, (ptitle, series) in enumerate(panels):
        x0 = left + k * (pw + gap)
        pts = [p for _, ps in series for p in ps]
        rlo = math.floor(min(r for r, _, _ in pts) * 10) / 10
        qlo = 10 ** math.floor(math.log10(min(q for _, q, _ in pts)))
        qhi = 10 ** math.ceil(math.log10(max(q for _, q, _ in pts)))
        tx = lambda r: x0 + pw * (r - rlo) / (1 - rlo)
        ty = lambda q: top + ph * (1 - (math.log10(q) - math.log10(qlo)) / (math.log10(qhi) - math.log10(qlo)))
        out.append(f'<text x="{x0}" y="{top - 14}" font-size="13" font-weight="600" fill="{TEXT}">{esc(ptitle)}</text>')
        t = qlo
        while t <= qhi * 1.001:
            for m in (1, 2, 5):
                v = t * m
                if v > qhi * 1.001:
                    break
                y = ty(v)
                out.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + pw:.1f}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
                if k == 0 or m == 1:
                    out.append(f'<text x="{x0 - 6}" y="{y + 4:.1f}" font-size="12" fill="{TEXT2}" text-anchor="end">{v:,.0f}</text>')
            t *= 10
        steps = int(round((1 - rlo) / 0.1))
        for i in range(steps + 1):
            r = rlo + i * 0.1
            out.append(f'<text x="{tx(r):.1f}" y="{top + ph + 20}" font-size="12" fill="{TEXT2}" text-anchor="middle">{r:.1f}</text>')
        out.append(f'<text x="{x0 + pw / 2:.1f}" y="{h - 30}" font-size="12" fill="{TEXT2}" text-anchor="middle">recall@10</text>')
        for i, (name, ps) in enumerate(series):
            ps = sorted(ps)
            line = " ".join(f"{tx(r):.1f},{ty(q):.1f}" for r, q, _ in ps)
            out.append(f'<polyline points="{line}" fill="none" stroke="{SERIES[i]}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
            for r, q, param in ps:
                out.append(f'<circle cx="{tx(r):.1f}" cy="{ty(q):.1f}" r="4" fill="{SERIES[i]}" stroke="{SURFACE}" stroke-width="2"><title>{esc(name)}, {esc(param)}: recall {r:.4f}, {q:,.0f} queries/s</title></circle>')
    out.append(f'<text x="16" y="{top + ph / 2}" font-size="12" fill="{TEXT2}" text-anchor="middle" transform="rotate(-90 16 {top + ph / 2})">queries/s, 1 thread (log)</text>')
    out.append(f'<text x="24" y="{h - 10}" font-size="12" fill="{TEXT2}">{esc(note)}</text>')
    out.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(out))


def main():
    os.makedirs("docs", exist_ok=True)
    note = "SmolLM2-360M-Instruct, bf16 weights; 4 vCPUs (Xeon, AVX-512). Replies of fixed length."
    p = {r["label"]: r for r in load("policies.jsonl")}
    bars(
        "docs/throughput.svg",
        "Throughput: the same 64 requests, three ways",
        "Output tokens per second; prompts of 32-256 tokens, replies of 16-256, all queued at once",
        [
            ("One at a time", p["sequential"]["output_tok_s"], False),
            ("Static batching", p["static"]["output_tok_s"], False),
            ("Continuous batching", p["continuous"]["output_tok_s"], True),
        ],
        "tok/s",
        note=note,
    )
    runs = load("load.jsonl")
    rates = sorted({float(r["label"].split("@")[1]) for r in runs})
    by = {r["label"]: r for r in runs}
    series = [
        (name, [by[f"{name}@{x:g}"]["ttft_p90_s"] for x in rates])
        for name in ("continuous", "static")
    ]
    lines(
        "docs/latency.svg",
        "Time to first token under load (90th percentile)",
        "Poisson arrivals, 48 requests per point; continuous vs static batching, 32 sequences max",
        rates,
        [("Continuous batching", series[0][1]), ("Static batching", series[1][1])],
        "arrival rate (requests per second)",
        "seconds",
        yfmt="{:.3g}",
        log=True,
        unit="s",
    )
    s = {r["label"]: r for r in load("spec.jsonl")}
    ks = [k for k in (2, 3, 4, 5, 6) if f"1.7b k={k}" in s]
    rows = [("No draft", s["1.7b k=0"]["output_tok_s"], False)]
    best = max(ks, key=lambda k: s[f"1.7b k={k}"]["output_tok_s"])
    for k in ks:
        r = s[f"1.7b k={k}"]
        acc = 100 * r["spec_accepted"] / max(r["spec_proposed"], 1)
        rows.append((f"k = {k} ({acc:.0f}% accepted)", r["output_tok_s"], k == best))
    bars(
        "docs/speculative.svg",
        "Speculative decoding: SmolLM2-1.7B, with SmolLM2-135M drafting",
        "Output tokens per second, one request at a time, 6 real prompts, up to 128 tokens each",
        rows,
        "tok/s",
        fmt="{:.1f}",
        note="Greedy decoding: the text is identical with and without the draft. k = tokens proposed per step.",
    )

    ann = load("ann.jsonl")

    def curve(engine, index, keep=lambda p: True):
        return [(r["recall10"], r["qps_1"], r["param"]) for r in ann
                if r["engine"] == engine and r["index"] == index and keep(r["param"])]

    refined = lambda p: p.endswith("refine=10")
    tradeoff(
        "docs/ann.svg",
        "Vector search on SIFT1M: ferrolm against FAISS",
        "1M 128-d vectors; HNSW M=16, efC=200; IVF-PQ 1,024 lists, 16-byte codes, top 100 re-ranked",
        [
            ("HNSW, ef 16 to 256", [("ferrolm", curve("ferrolm", "hnsw")), ("FAISS 1.15", curve("faiss", "hnsw"))]),
            ("IVF-PQ with re-ranking, nprobe 1 to 64", [
                ("ferrolm", curve("ferrolm", "ivfpq", refined)), ("FAISS 1.15", curve("faiss", "ivfpq", refined))]),
        ],
        "Up and to the right is better. One thread of a 4-vCPU Xeon (AVX-512); 2,000 of the 10,000 queries timed.",
    )
    if os.path.exists("bench/results/qa.jsonl"):
        qa = {r["mode"]: r for r in load("qa.jsonl")}
        names = {"closed": "No retrieval", "rag": "Retrieved passages", "oracle": "Gold passage"}
        n = next(iter(qa.values()))
        bars(
            "docs/rag.svg",
            "Question answering with and without retrieval",
            f"SQuAD v1.1 dev, {n['n']} questions; F1 of the model's short answers",
            [(names[m], qa[m]["f1"], m == "rag") for m in ("closed", "rag", "oracle") if m in qa],
            "F1",
            fmt="{:.1f}",
            note=f"SmolLM2-1.7B-Instruct answers; bge-small retrieves the top {n['k']} of all 2,067 distinct dev paragraphs.",
        )


if __name__ == "__main__":
    main()

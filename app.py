import base64
import csv
import html
import io
import math
from collections import Counter
from pathlib import Path

import joblib
import streamlit as st

# =========================================================
# PAGE
# =========================================================
st.set_page_config(
    page_title="NEXA DNA",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MIN_LENGTH = 5
VALID_BASES = set("ACGT")
BASE_COLORS = {"A": "#39f0d8", "C": "#6f8bff", "G": "#b86bff", "T": "#ff5fa8"}

SAMPLES = {
    "⚡ Sample 1": "ATGCGTACGTAGCTAGCTAGGCTAATCGATCGGATCCGATAGCTAGCTAAGCT",
    "🌊 Sample 2": "GGCCGGCCATATGCGCGCTATAGCGCGCCGGATATATCGCGCGGCCTA",
    "🔮 Sample 3": "TTAGGGTTAGGGTTAGGGATCGATCGTTAACCGGTTAACCATGCATGC",
}


@st.cache_resource(show_spinner="Booting the model…")
def load_artifacts():
    model = joblib.load(BASE_DIR / "models" / "dna_model.pkl")
    vectorizer = joblib.load(BASE_DIR / "models" / "dna_vectorizer.pkl")
    return model, vectorizer


try:
    model, vectorizer = load_artifacts()
except FileNotFoundError as err:
    st.error(
        f"Model file not found: `{Path(err.filename).name}`. "
        "Put `dna_model.pkl` and `dna_vectorizer.pkl` inside a `models/` "
        "folder next to this file."
    )
    st.stop()


# =========================================================
# STYLE
# =========================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
    --bg: #070614;
    --glass: rgba(20, 18, 48, 0.55);
    --edge: rgba(150, 140, 255, 0.22);
    --text: #f1eefe;
    --muted: #a39fc9;
    --cyan: #39f0d8;
    --blue: #6f8bff;
    --violet: #b86bff;
    --pink: #ff5fa8;
}

html, body, .stApp, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text);
    font-family: 'Sora', sans-serif;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none; }
.block-container { max-width: 820px; padding-top: 2rem; padding-bottom: 5rem; position: relative; z-index: 2; }

/* ---------- aurora + grid background ---------- */
.stApp::before {
    content: "";
    position: fixed; inset: -20%;
    z-index: 0; pointer-events: none;
    background:
        radial-gradient(40% 40% at 20% 20%, rgba(57, 240, 216, .28), transparent 70%),
        radial-gradient(45% 45% at 80% 25%, rgba(184, 107, 255, .30), transparent 70%),
        radial-gradient(40% 40% at 60% 85%, rgba(255, 95, 168, .22), transparent 70%);
    filter: blur(30px);
    animation: drift 18s ease-in-out infinite alternate;
}
.stApp::after {
    content: "";
    position: fixed; inset: 0;
    z-index: 1; pointer-events: none;
    background-image:
        linear-gradient(rgba(255,255,255,.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,.035) 1px, transparent 1px);
    background-size: 46px 46px;
    mask-image: radial-gradient(ellipse at center, black 30%, transparent 80%);
    -webkit-mask-image: radial-gradient(ellipse at center, black 30%, transparent 80%);
}
@keyframes drift {
    from { transform: translate3d(-3%, -2%, 0) scale(1); }
    to   { transform: translate3d(3%, 3%, 0) scale(1.12); }
}

/* ---------- top pill ---------- */
.topbar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 3rem; }
.logo { font-weight: 800; font-size: 1.15rem; letter-spacing: -0.02em; }
.logo b { background: linear-gradient(90deg, var(--cyan), var(--violet)); -webkit-background-clip: text; background-clip: text; color: transparent; }
.pill {
    display: inline-flex; align-items: center; gap: .5rem;
    padding: .4rem .9rem; border-radius: 999px;
    background: var(--glass); border: 1px solid var(--edge);
    backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
    font-family: 'JetBrains Mono', monospace; font-size: .75rem; color: var(--cyan);
}
.dot { width: 8px; height: 8px; border-radius: 50%; background: var(--cyan); box-shadow: 0 0 10px var(--cyan); animation: pulse 1.8s infinite; }
@keyframes pulse { 50% { opacity: .35; transform: scale(.75); } }

/* ---------- hero ---------- */
.kicker { font-family: 'JetBrains Mono', monospace; font-size: .8rem; color: var(--cyan); letter-spacing: .12em; }
.hero {
    font-size: clamp(2.6rem, 8vw, 4.8rem); font-weight: 800; line-height: 1.02;
    letter-spacing: -0.045em; margin: .6rem 0 1rem;
    background: linear-gradient(100deg, #ffffff 0%, var(--cyan) 35%, var(--violet) 70%, var(--pink) 100%);
    background-size: 200% auto;
    -webkit-background-clip: text; background-clip: text; color: transparent;
    animation: shine 8s linear infinite;
}
@keyframes shine { to { background-position: 200% center; } }
.lede { color: var(--muted); font-size: 1.05rem; line-height: 1.7; max-width: 54ch; }

/* ---------- DNA marquee ---------- */
.marquee { overflow: hidden; margin: 2rem 0 2.4rem; mask-image: linear-gradient(90deg, transparent, black 12%, black 88%, transparent); -webkit-mask-image: linear-gradient(90deg, transparent, black 12%, black 88%, transparent); }
.track-m { display: inline-flex; gap: 1.1rem; white-space: nowrap; animation: scroll 28s linear infinite; font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 1.3rem; }
.track-m span:nth-child(4n+1) { color: var(--cyan); text-shadow: 0 0 12px var(--cyan); }
.track-m span:nth-child(4n+2) { color: var(--blue); text-shadow: 0 0 12px var(--blue); }
.track-m span:nth-child(4n+3) { color: var(--violet); text-shadow: 0 0 12px var(--violet); }
.track-m span:nth-child(4n+4) { color: var(--pink); text-shadow: 0 0 12px var(--pink); }
@keyframes scroll { to { transform: translateX(-50%); } }

/* ---------- glass input ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--glass) !important;
    border: 1px solid var(--edge) !important;
    border-radius: 24px !important;
    backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
    box-shadow: 0 20px 80px rgba(80, 60, 200, .18);
    padding: .4rem;
}
label, [data-testid="stWidgetLabel"] p { color: var(--text) !important; font-weight: 600 !important; }
textarea {
    background: rgba(5, 4, 20, .75) !important;
    color: #eafffb !important;
    border: 1px solid var(--edge) !important;
    border-radius: 16px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: .95rem !important; line-height: 1.7 !important;
}
textarea:focus { border-color: var(--cyan) !important; box-shadow: 0 0 0 1px var(--cyan), 0 0 28px rgba(57, 240, 216, .25) !important; }

/* ---------- buttons ---------- */
.stButton > button {
    border-radius: 14px !important; border: 1px solid var(--edge) !important;
    background: rgba(255, 255, 255, .04) !important; color: var(--text) !important;
    font-family: 'Sora', sans-serif !important; font-weight: 600 !important;
    transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
}
.stButton > button:hover { border-color: var(--cyan) !important; color: var(--cyan) !important; transform: translateY(-2px); }
.stButton > button[kind="primary"] {
    height: 3.6rem; border: none !important; font-size: 1rem !important; font-weight: 800 !important;
    background: linear-gradient(100deg, var(--blue), var(--violet) 50%, var(--pink)) !important;
    color: #fff !important;
    box-shadow: 0 12px 40px rgba(150, 90, 255, .45);
}
.stButton > button[kind="primary"]:hover { color: #fff !important; transform: translateY(-3px); box-shadow: 0 18px 55px rgba(255, 95, 168, .45); }
.stButton > button:focus-visible { outline: 2px solid var(--cyan) !important; outline-offset: 3px; }

/* ---------- composition strip ---------- */
.strip { display: flex; height: 12px; border-radius: 99px; overflow: hidden; margin: 2.2rem 0 1.4rem; box-shadow: 0 0 30px rgba(184, 107, 255, .35); }
.strip span { display: block; height: 100%; }
.legend { display: flex; gap: 1.2rem; font-family: 'JetBrains Mono', monospace; font-size: .78rem; color: var(--muted); margin: -.6rem 0 1.6rem; }
.legend i { display: inline-block; width: 9px; height: 9px; border-radius: 3px; margin-right: .4rem; }

/* ---------- result ---------- */
.result {
    position: relative; border-radius: 26px; padding: 2rem 2rem 1.8rem; margin: 1rem 0 1.2rem;
    background: linear-gradient(160deg, rgba(30, 26, 72, .8), rgba(12, 10, 34, .85));
    backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px);
}
.result::before {
    content: ""; position: absolute; inset: 0; border-radius: 26px; padding: 1.5px;
    background: linear-gradient(120deg, var(--cyan), var(--violet), var(--pink));
    -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
    -webkit-mask-composite: xor; mask-composite: exclude; pointer-events: none;
}
.result .label { font-family: 'JetBrains Mono', monospace; font-size: .78rem; color: var(--muted); letter-spacing: .08em; }
.result .family {
    font-size: clamp(2rem, 6vw, 3.2rem); font-weight: 800; letter-spacing: -0.03em; margin: .4rem 0 .7rem;
    background: linear-gradient(100deg, var(--cyan), var(--violet)); -webkit-background-clip: text; background-clip: text; color: transparent;
}
.result .conf { font-family: 'JetBrains Mono', monospace; color: var(--text); }
.result .conf b { color: var(--cyan); text-shadow: 0 0 14px rgba(57, 240, 216, .6); }

.stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: .9rem; margin-bottom: 2.4rem; }
.stat { background: var(--glass); border: 1px solid var(--edge); border-radius: 18px; padding: 1rem 1.1rem; backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px); }
.stat .k { color: var(--muted); font-size: .78rem; }
.stat .v { font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 1.3rem; margin-top: .25rem; color: var(--cyan); }

/* ---------- probability bars ---------- */
.sec { font-size: 1.5rem; font-weight: 800; letter-spacing: -0.03em; margin: 0 0 1rem; }
.row { display: grid; grid-template-columns: minmax(90px, 190px) 1fr 66px; gap: .9rem; align-items: center; margin: .75rem 0; }
.row .name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-weight: 600; }
.row .bar { background: rgba(255, 255, 255, .06); border-radius: 99px; height: 12px; overflow: hidden; }
.row .fill {
    height: 100%; border-radius: 99px;
    background: linear-gradient(90deg, var(--blue), var(--violet));
    transform-origin: left; animation: grow .9s cubic-bezier(.2, .8, .2, 1) both;
}
.row.top .fill { background: linear-gradient(90deg, var(--cyan), var(--blue), var(--pink)); box-shadow: 0 0 18px rgba(57, 240, 216, .6); }
.row .pct { text-align: right; font-family: 'JetBrains Mono', monospace; color: var(--muted); font-size: .9rem; }
.row.top .pct { color: var(--cyan); font-weight: 700; }
@keyframes grow { from { transform: scaleX(0); } }

.feat { display: grid; grid-template-columns: repeat(3, 1fr); gap: .9rem; margin-top: 3rem; }
.feat div { background: var(--glass); border: 1px solid var(--edge); border-radius: 18px; padding: 1.1rem; font-size: .88rem; color: var(--muted); line-height: 1.55; }
.feat b { display: block; color: var(--text); font-size: 1rem; margin-bottom: .3rem; }

/* ---------- helix images ---------- */
.herowrap { display: flex; align-items: center; justify-content: space-between; gap: 1rem; }
.herotext { flex: 1; min-width: 0; }
.helix-v { width: 200px; height: 400px; flex: none; filter: drop-shadow(0 0 28px rgba(184, 107, 255, .5)); }
.helix-h { width: 100%; height: 110px; display: block; margin: 2.6rem 0 0; filter: drop-shadow(0 0 22px rgba(57, 240, 216, .35)); }
.helix-s { position: absolute; right: 1.6rem; top: 50%; transform: translateY(-50%); width: 90px; height: 170px; filter: drop-shadow(0 0 16px rgba(184, 107, 255, .5)); }
.result { padding-right: 8rem; }

/* ---------- tabs, uploader, download, chips ---------- */
.stTabs [data-baseweb="tab-list"] { gap: .4rem; border-bottom: 1px solid var(--edge); }
.stTabs [data-baseweb="tab"] { color: var(--muted) !important; font-weight: 600; border-radius: 10px 10px 0 0; }
.stTabs [aria-selected="true"] { color: var(--cyan) !important; }
.stTabs [data-baseweb="tab-highlight"] { background: linear-gradient(90deg, var(--cyan), var(--violet)) !important; height: 3px !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.4rem; }
[data-testid="stFileUploaderDropzone"] { background: rgba(5, 4, 20, .6) !important; border: 1px dashed var(--edge) !important; border-radius: 16px !important; }
[data-testid="stFileUploaderDropzone"] * { color: var(--muted) !important; }
[data-testid="stFileUploaderDropzone"] button { color: var(--text) !important; }
.stDownloadButton > button { border-radius: 14px !important; border: 1px solid var(--cyan) !important; background: rgba(57, 240, 216, .08) !important; color: var(--cyan) !important; font-weight: 700 !important; height: 3rem; }
.stDownloadButton > button:hover { background: rgba(57, 240, 216, .18) !important; box-shadow: 0 0 24px rgba(57, 240, 216, .3); }
.chips { display: flex; flex-wrap: wrap; gap: .5rem; }
.chip { padding: .4rem .9rem; border-radius: 99px; border: 1px solid var(--edge); background: var(--glass); font-family: 'JetBrains Mono', monospace; font-size: .82rem; color: var(--text); }

.foot { text-align: center; color: var(--muted); font-size: .8rem; margin-top: 3rem; }

@media (max-width: 640px) {
    .stats, .feat { grid-template-columns: 1fr; }
    .topbar { margin-bottom: 2rem; }
    .helix-v { width: 90px; height: 200px; }
    .helix-s { display: none; }
    .result { padding-right: 2rem; }
    .helix-h { height: 70px; }
}
@media (prefers-reduced-motion: reduce) {
    .stApp::before, .hero, .track-m, .dot, .row .fill { animation: none !important; }
}
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# HELPERS
# =========================================================
def clean_sequence(raw: str) -> str:
    """Drop FASTA headers, whitespace and digits; uppercase the rest."""
    lines = [ln for ln in raw.splitlines() if not ln.strip().startswith(">")]
    text = "".join(lines).upper()
    return "".join(ch for ch in text if not ch.isspace() and not ch.isdigit())


def composition_strip(seq: str) -> str:
    counts, total = Counter(seq), len(seq) or 1
    parts = "".join(
        f'<span style="width:{counts.get(b, 0) / total * 100:.2f}%;background:{c}"></span>'
        for b, c in BASE_COLORS.items()
    )
    legend = "".join(
        f'<span><i style="background:{c}"></i>{b} {counts.get(b, 0) / total * 100:.0f}%</span>'
        for b, c in BASE_COLORS.items()
    )
    return f'<div class="strip">{parts}</div><div class="legend">{legend}</div>'


def load_sample(text: str):
    st.session_state["sequence"] = text


def helix_uri(w, h, period, horizontal=False, per_period=8, speed=9, amp_ratio=0.32):
    """Build an animated DNA double helix as an inline SVG data URI (no external files)."""
    length = w if horizontal else h
    cross = h if horizontal else w
    c, amp = cross / 2, cross * amp_ratio

    def pt(t, off):
        return (t, off) if horizontal else (off, t)

    # backbone strands
    fine = period / 40
    steps = int((length + 2 * period) / fine) + 1
    s1, s2 = [], []
    for i in range(steps):
        t = -period + i * fine
        s = math.sin(2 * math.pi * t / period)
        x1, y1 = pt(t, c + amp * s)
        x2, y2 = pt(t, c - amp * s)
        s1.append(f"{x1:.1f},{y1:.1f}")
        s2.append(f"{x2:.1f},{y2:.1f}")

    # base-pair rungs and glowing nodes
    gap = period / per_period
    n = int((length + 2 * period) / gap) + 1
    pairs = [("#39f0d8", "#ff5fa8"), ("#6f8bff", "#b86bff")]
    rungs, nodes = [], []
    for i in range(n):
        t = -period + i * gap
        th = 2 * math.pi * t / period
        s, d = math.sin(th), math.cos(th)
        k = (d + 1) / 2
        c1, c2 = pairs[i % 2]
        x1, y1 = pt(t, c + amp * s)
        x2, y2 = pt(t, c - amp * s)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        op = 0.25 + 0.4 * k
        rungs.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{mx:.1f}" y2="{my:.1f}" stroke="{c1}" stroke-width="2.4" stroke-linecap="round" opacity="{op:.2f}"/>'
            f'<line x1="{mx:.1f}" y1="{my:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{c2}" stroke-width="2.4" stroke-linecap="round" opacity="{op:.2f}"/>'
        )
        nodes.append((d, f'<circle cx="{x1:.1f}" cy="{y1:.1f}" r="{4 + 3.5 * k:.1f}" fill="{c1}" opacity="{0.5 + 0.5 * k:.2f}"/>'))
        nodes.append((-d, f'<circle cx="{x2:.1f}" cy="{y2:.1f}" r="{4 + 3.5 * (1 - k):.1f}" fill="{c2}" opacity="{0.5 + 0.5 * (1 - k):.2f}"/>'))
    nodes.sort(key=lambda x: x[0])

    shift = f"translate({period}px,0)" if horizontal else f"translate(0,{period}px)"
    grad = (
        '<linearGradient id="f" x1="0" y1="0" x2="1" y2="0">'
        if horizontal
        else '<linearGradient id="f" x1="0" y1="0" x2="0" y2="1">'
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">
<style>.m{{animation:mv {speed}s linear infinite}}@keyframes mv{{to{{transform:{shift}}}}}@media(prefers-reduced-motion:reduce){{.m{{animation:none}}}}</style>
<defs>{grad}<stop offset="0" stop-color="#000"/><stop offset=".18" stop-color="#fff"/><stop offset=".82" stop-color="#fff"/><stop offset="1" stop-color="#000"/></linearGradient>
<mask id="k"><rect width="{w}" height="{h}" fill="url(#f)"/></mask>
<filter id="g" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="2.2" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>
<g mask="url(#k)"><g filter="url(#g)"><g class="m">
<polyline points="{' '.join(s1)}" fill="none" stroke="#39f0d8" stroke-width="2.6" opacity=".55"/>
<polyline points="{' '.join(s2)}" fill="none" stroke="#ff5fa8" stroke-width="2.6" opacity=".55"/>
{''.join(rungs)}{''.join(n_[1] for n_ in nodes)}
</g></g></g></svg>"""
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


HELIX_V = helix_uri(200, 400, 200)                    # hero helix
HELIX_H = helix_uri(1200, 110, 220, horizontal=True)  # wide divider helix
HELIX_S = helix_uri(90, 170, 110, speed=7)            # small helix for the result card


# =========================================================
# TOP BAR + HERO
# =========================================================
st.markdown(
    '<div class="topbar"><div class="logo">🧬 NEXA <b>DNA</b></div>'
    '<div class="pill"><span class="dot"></span>MODEL ONLINE</div></div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="herowrap"><div class="herotext">'
    '<div class="kicker">AI × GENOMICS × MACHINE LEARNING</div>'
    '<div class="hero">Decode the unknown.</div>'
    '<p class="lede">Drop in a nucleotide sequence and the model tells you which '
    "gene family it belongs to, plus how sure it is about every class.</p>"
    f'</div><img class="helix-v" src="{HELIX_V}" alt=""></div>',
    unsafe_allow_html=True,
)

ticker = "".join(f"<span>{b}</span>" for b in "ATGCAGCTGCATTAGC" * 6)
st.markdown(
    f'<div class="marquee"><div class="track-m">{ticker}{ticker}</div></div>',
    unsafe_allow_html=True,
)

# =========================================================
# TABS
# =========================================================
tab_analyze, tab_about = st.tabs(["🧬 Analyze", "📊 About the model"])

# ---------------------------------------------------------
# ANALYZE TAB
# ---------------------------------------------------------
with tab_analyze:
    with st.container(border=True):
        sequence_input = st.text_area(
            "DNA sequence",
            key="sequence",
            placeholder="ATGCGTACGTAGCTAGCTAG…  (FASTA works too)",
            height=170,
        )
        uploaded = st.file_uploader(
            "Or upload a FASTA / text file",
            type=["fasta", "fa", "fna", "txt"],
        )
        cols = st.columns(len(SAMPLES))
        for col, (name, text) in zip(cols, SAMPLES.items()):
            col.button(name, on_click=load_sample, args=(text,), use_container_width=True)

        analyze = st.button("🧬  Analyze sequence", type="primary", use_container_width=True)

    if analyze:
        st.session_state.pop("result", None)
        raw = uploaded.getvalue().decode("utf-8", errors="ignore") if uploaded else sequence_input
        seq = clean_sequence(raw)
        invalid = sorted(set(seq) - VALID_BASES)

        if not seq:
            st.warning("Paste a DNA sequence or upload a file first.")
        elif invalid:
            st.error(f"Invalid characters found: {', '.join(invalid)}. Only A, C, G and T are allowed.")
        elif len(seq) < MIN_LENGTH:
            st.warning(f"Too short. Enter at least {MIN_LENGTH} bases.")
        else:
            probabilities = model.predict_proba(vectorizer.transform([seq]))[0]
            st.session_state["result"] = {
                "seq": seq,
                "classes": [str(c) for c in model.classes_],
                "probs": [float(p) for p in probabilities],
            }

    # results live in session_state so the download button doesn't wipe them
    res = st.session_state.get("result")
    if res:
        seq, classes, probs = res["seq"], res["classes"], res["probs"]
        top = max(range(len(probs)), key=probs.__getitem__)
        prediction, confidence = classes[top], probs[top] * 100
        gc = (seq.count("G") + seq.count("C")) / len(seq) * 100

        st.markdown(composition_strip(seq), unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="result">
                <img class="helix-s" src="{HELIX_S}" alt="">
                <div class="label">PREDICTED GENE FAMILY</div>
                <div class="family">{html.escape(prediction)}</div>
                <div class="conf">Model confidence <b>{confidence:.2f}%</b></div>
            </div>
            <div class="stats">
                <div class="stat"><div class="k">Length</div><div class="v">{len(seq):,} bp</div></div>
                <div class="stat"><div class="k">GC content</div><div class="v">{gc:.1f}%</div></div>
                <div class="stat"><div class="k">Classes</div><div class="v">{len(classes)}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if confidence < 50:
            st.info("The model isn't very sure about this one. Treat the result as a weak signal.")

        rows = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
        bars = "".join(
            f'<div class="row{" top" if i == 0 else ""}">'
            f'<div class="name" title="{html.escape(fam)}">{html.escape(fam)}</div>'
            f'<div class="bar"><div class="fill" style="width:{p * 100:.2f}%"></div></div>'
            f'<div class="pct">{p * 100:.1f}%</div></div>'
            for i, (fam, p) in enumerate(rows)
        )
        st.markdown(f'<div class="sec">What the model sees</div>{bars}', unsafe_allow_html=True)

        # downloadable report
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["predicted_family", prediction])
        writer.writerow(["confidence_percent", f"{confidence:.2f}"])
        writer.writerow(["length_bp", len(seq)])
        writer.writerow(["gc_content_percent", f"{gc:.1f}"])
        writer.writerow([])
        writer.writerow(["family", "probability_percent"])
        for fam, p in rows:
            writer.writerow([fam, f"{p * 100:.2f}"])

        st.write("")
        st.download_button(
            "⬇  Download results (CSV)",
            data=buf.getvalue(),
            file_name="nexa_dna_prediction.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ---------------------------------------------------------
# ABOUT TAB
# ---------------------------------------------------------
with tab_about:
    class_list = [str(c) for c in model.classes_]
    try:
        n_features = len(vectorizer.get_feature_names_out())
    except Exception:
        n_features = "n/a"
    ngram = getattr(vectorizer, "ngram_range", None)
    ngram_text = f"{ngram[0]} to {ngram[1]}" if ngram else "n/a"
    chips = "".join(f'<span class="chip">{html.escape(c)}</span>' for c in class_list)

    st.markdown(
        f"""
        <div class="sec">Gene families it knows</div>
        <div class="chips">{chips}</div>
        <div class="stats" style="margin-top:1.6rem">
            <div class="stat"><div class="k">Classes</div><div class="v">{len(class_list)}</div></div>
            <div class="stat"><div class="k">Features</div><div class="v">{n_features}</div></div>
            <div class="stat"><div class="k">k-mer size</div><div class="v">{ngram_text}</div></div>
        </div>
        <div class="sec">How it works</div>
        <div class="feat" style="margin-top:.4rem">
            <div><b>🧬 1. Read</b>Your sequence is cleaned and checked for A, C, G and T only.</div>
            <div><b>🧠 2. Convert</b>Character-level TF-IDF turns short DNA patterns into numbers.</div>
            <div><b>⚡ 3. Predict</b>Logistic Regression scores every gene family and picks the highest.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# FOOTER
# =========================================================
st.markdown(
    f"""
    <img class="helix-h" src="{HELIX_H}" alt="">
    <div class="foot">NEXA DNA &middot; Built with Python, Scikit-learn &amp; Streamlit &middot; Synthetic dataset for educational purposes</div>
    """,
    unsafe_allow_html=True,
)
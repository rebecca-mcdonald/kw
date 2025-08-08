# LBC Question-First Keyword App (Streamlit MVP)
# -------------------------------------------------
# One-file Streamlit prototype that turns a URL + seed keywords
# into a question-first keyword map, content briefs, and FAQ JSON-LD.
#
# ▶ How to run locally
#   1) pip install -r requirements.txt  (see list below)
#   2) streamlit run app.py
#
# ▶ Deploy options
#   • Streamlit Community Cloud: push to GitHub, click "New app".
#   • Hugging Face Spaces (Streamlit template): add this file as app.py.
#
# ▶ Optional integrations (placeholders included below)
#   • SERP provider (autocomplete/PAA/related): set env VARS and toggle in UI
#   • Volumes/Difficulty provider: set API key and enable in UI
#   • GSC import: upload CSV export (queries/pages) and we’ll mine questions
#
# requirements.txt (put these in a file when you deploy)
# -----------------------------------------------------
# streamlit==1.37.0
# pydantic==2.8.2
# pandas==2.2.2
# scikit-learn==1.4.2
# numpy==1.26.4
# plotly==5.22.0
# requests==2.32.2
# python-dotenv==1.0.1
#
# NOTE: This app uses placeholder scoring and clustering logic so you can demo end-to-end.
# Swap in your providers where marked with TODOs.

import os
import io
import json
import time
import base64
from typing import List, Dict, Any

import pandas as pd
import numpy as np
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

APP_TITLE = "Question‑First Keyword Map (MVP)"
DEFAULT_SEEDS = "title 24 ja8, led wall pack, troffer 2x4, recessed downlight commercial, 0-10v dimming"

# -----------------------------
# Models / schema
# -----------------------------
class AnalyzeRequest(BaseModel):
    project_url: str
    seeds: List[str]
    geography: str = "US"
    business_value_overrides: Dict[str, float] | None = None

# -----------------------------
# Utilities
# -----------------------------

def band_value(band: str, kind: str = "volume") -> float:
    maps = {
        "volume": {"High": 0.9, "Med–High": 0.75, "Med": 0.6, "Low–Med": 0.4, "Low": 0.25},
        "difficulty": {"Low": 0.85, "Low–Med": 0.75, "Med": 0.6, "Med–High": 0.45, "High": 0.3},
    }
    return maps[kind].get(band, 0.6)


def compute_os(volume_band: str, difficulty_band: str, geography: str = "US", sitegap: float = 0.5,
               business_value: float = 0.5, serp_potential: float = 0.5) -> float:
    vol_w = band_value(volume_band, "volume")
    diff_w = band_value(difficulty_band, "difficulty")
    geo_w = 0.8 if geography == "US" else 0.6
    os = 0.25*vol_w + 0.15*diff_w + 0.25*sitegap + 0.10*geo_w + 0.20*business_value + 0.05*serp_potential
    return round(float(os), 3)


def df_to_csv_download(df: pd.DataFrame, filename: str) -> str:
    csv = df.to_csv(index=False).encode("utf-8")
    b64 = base64.b64encode(csv).decode()
    return f'<a download="{filename}" href="data:text/csv;base64,{b64}">Download CSV</a>'

# -----------------------------
# Placeholder question generator
# -----------------------------

SEED_TEMPLATES: Dict[str, List[str]] = {
    "title 24 ja8": [
        "What is Title 24 lighting (California) and who must comply?",
        "Which LED downlights are JA8 compliant (2024/2025 lists)?",
        "How to pass a Title 24 lighting inspection (checklist)?",
        "Is LED tape lighting eligible for JA8 residential projects?",
    ],
    "led wall pack": [
        "What wattage LED wall pack replaces 250W metal halide?",
        "Cutoff vs full‑cutoff wall packs—light pollution rules",
        "Photocell and 0–10V dimming on wall packs—how to wire?",
        "Dark‑Sky friendly wall packs—what to look for?",
    ],
    "troffer 2x4": [
        "2x2 vs 2x4 LED troffer—when to use each?",
        "Back‑lit vs edge‑lit troffers—pros and cons",
        "What is a lay‑in vs surface‑mount troffer?",
    ],
    "recessed downlight commercial": [
        "Best commercial recessed LED housings for offices (specs)",
        "4\" vs 6\" recessed downlights—when to use each?",
        "Canless vs housing recessed lights for new construction",
        "Flanged vs flangeless trims—what’s the difference?",
    ],
    "0-10v dimming": [
        "0–10V vs TRIAC dimming—what’s the difference?",
        "Do I need an ELV dimmer for LED fixtures?",
        "How to troubleshoot flicker with 0–10V drivers",
    ],
}

INTENT_RULES = {
    "how": "Informational",
    "what": "Informational",
    "which": "Informational",
    "best": "Commercial",
    "vs": "Commercial",
    "price": "Commercial",
    "buy": "Transactional",
    "shipping": "Transactional",
}

PAGE_TYPE_RULES = {
    "Title 24": "Guide + Product hub + FAQ",
    "wall pack": "Buying guide + How‑to",
    "troffer": "Guide + Calculator + Product hub",
    "recessed": "Category pillar + Comparison",
    "dimming": "Troubleshooting guide + FAQ",
}

# -----------------------------
# Optional: SERP provider hooks (stubs)
# -----------------------------
SERP_API_KEY = os.getenv("SERP_API_KEY", "")
VOLUME_API_KEY = os.getenv("VOLUME_API_KEY", "")


def serp_expand(seed: str) -> List[str]:
    """TODO: Replace with real autocomplete/PAA/related via your SERP provider.
    For demo, we return the templates plus simple variations."""
    base = SEED_TEMPLATES.get(seed.lower().strip(), [])
    variants = [q.replace("?", "").lower() for q in base]
    # Fake expansion by adding question stems
    extra = [f"why {v}" for v in variants[:1]] + [f"does {v}" for v in variants[1:2]]
    return base + [e.capitalize() + "?" for e in extra]


def estimate_bands(question: str) -> tuple[str, str]:
    """Placeholder band estimator. Swap with real volume/difficulty provider."""
    q = question.lower()
    if any(k in q for k in ["best", "vs", "replace", "2x4", "recessed"]):
        return "High", "Med–High"
    if any(k in q for k in ["title 24", "ja8", "0–10v", "0-10v", "photocell", "dark‑sky", "dark-sky"]):
        return "Med", "Med"
    return "Low–Med", "Low–Med"


def rules_page_type(question: str) -> str:
    for k, v in PAGE_TYPE_RULES.items():
        if k.lower() in question.lower():
            return v
    return "Guide"


def rules_intent(question: str) -> str:
    q = question.lower()
    if " buy" in q or "shipping" in q or "lead time" in q:
        return "Transactional"
    if " vs" in q or "best" in q:
        return "Commercial"
    for stem, label in INTENT_RULES.items():
        if q.startswith(stem + " "):
            return label
    return "Informational"

# -----------------------------
# Brief / JSON‑LD builders
# -----------------------------

def build_brief(cluster_label: str, questions: List[str], page_type: str, url_slug: str) -> str:
    h2 = [
        "What it is / why it matters",
        "Specs to compare (lumens, CCT, CRI, efficacy)",
        "How to choose (use‑case driven)",
        "Compliance & standards",
        "Wiring/controls or installation notes",
        "Recommended products",
        "FAQ"
    ]
    md = [
        f"# Content Brief: {cluster_label}",
        f"**Page Type:** {page_type}",
        f"**Proposed URL:** {url_slug}",
        "\n**Primary Questions**",
    ]
    md += [f"- {q}" for q in questions[:5]]
    md += ["\n## Outline (H2/H3)"] + [f"- {x}" for x in h2]
    md += ["\n## Internal Links\n- Relevant category pages\n- Controls/dimmers\n- Volume quote / Contact\n- Shipping & pickup policy"]
    md += ["\n## CTA\n- Get a volume quote\n- Request submittals/spec sheets"]
    return "\n".join(md)


def build_faq_jsonld(questions: List[str]) -> Dict[str, Any]:
    entities = [{
        "@type": "Question",
        "name": q,
        "acceptedAnswer": {"@type": "Answer", "text": "Short, helpful answer (90–160 words)."}
    } for q in questions[:5]]
    return {"@context": "https://schema.org", "@type": "FAQPage", "mainEntity": entities}

# -----------------------------
# App UI
# -----------------------------

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

st.markdown("Design an SEO/LLM‑SEO question map from a site + seed keywords, score them by opportunity, and export briefs/FAQs.")

with st.sidebar:
    st.header("Inputs")
    url = st.text_input("Client URL", placeholder="https://example.com")
    seeds_text = st.text_area("Seed keywords (comma or newline)", value=DEFAULT_SEEDS, height=110)
    geography = st.selectbox("Target geography", ["US", "CA", "Global"], index=0)
    use_serp = st.toggle("Use SERP expansion (demo stub)", value=True)
    use_volumes = st.toggle("Attach volume/difficulty (demo bands)", value=True)
    run_btn = st.button("Run Analysis", type="primary")

if run_btn:
    seeds = [s.strip() for s in seeds_text.replace("\n", ",").split(",") if s.strip()]
    req = AnalyzeRequest(project_url=url or "https://example.com", seeds=seeds, geography=geography)

    # 1) Generate questions per seed
    question_rows = []
    for seed in req.seeds:
        qs = serp_expand(seed) if use_serp else SEED_TEMPLATES.get(seed.lower(), [])
        for q in qs:
            vol_band, diff_band = estimate_bands(q) if use_volumes else ("Med", "Med")
            page_type = rules_page_type(q)
            intent = rules_intent(q)
            oscore = compute_os(vol_band, diff_band, geography=req.geography)
            question_rows.append({
                "Seed": seed,
                "Question": q,
                "Intent": intent,
                "Volume Band": vol_band,
                "Difficulty Band": diff_band,
                "Recommended Page Type": page_type,
                "On‑Site Coverage": "TBD",
                "Opportunity Score": oscore,
            })

    df = pd.DataFrame(question_rows)

    # 2) Lightweight clustering for labels
    if not df.empty:
        vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
        X = vec.fit_transform(df["Question"])  # noqa: F841
        n_clusters = min(8, max(2, len(df)//10))
        km = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
        labels = km.fit_predict(X)
        df["Cluster ID"] = labels

        # Label clusters by most frequent bigram/keyword
        cluster_labels = {}
        for cid in sorted(set(labels)):
            subset = df[df["Cluster ID"] == cid]
            # Heuristic: pick a representative token
            tokens = " ".join(subset["Question"]).lower().split()
            rep = None
            for k in ["title", "ja8", "wall", "troffer", "recessed", "dimming", "tape", "high", "track", "emergency"]:
                if k in tokens:
                    rep = k
                    break
            cluster_labels[cid] = rep or f"Topic {cid+1}"
        df["Cluster Label"] = df["Cluster ID"].map(cluster_labels)

    st.success("Analysis complete")

    # 3) Display table
    st.subheader("Questions Table")
    st.dataframe(df.sort_values(["Opportunity Score"], ascending=False), use_container_width=True)

    # 4) Briefs per cluster
    st.subheader("Content Briefs (per cluster)")
    briefs = []
    for label in sorted(df["Cluster Label"].unique()):
        subset = df[df["Cluster Label"] == label].sort_values("Opportunity Score", ascending=False)
        top_qs = subset["Question"].tolist()
        # Guess a page type from majority vote
        page_type = subset["Recommended Page Type"].mode().iloc[0]
        slug_token = label.replace(" ", "-")
        brief_md = build_brief(label, top_qs, page_type, f"/resources/{slug_token}-guide")
        briefs.append((label, brief_md, top_qs[:5]))

    for label, md, top_qs in briefs:
        with st.expander(f"Brief: {label}"):
            st.markdown(md)
            faq_json = build_faq_jsonld(top_qs)
            st.code(json.dumps(faq_json, indent=2), language="json")

    # 5) Downloads
    st.subheader("Export")
    st.markdown(df_to_csv_download(df, "question_map.csv"), unsafe_allow_html=True)

    # All briefs to a single markdown
    all_md = "\n\n".join([b[1] for b in briefs])
    st.download_button("Download Briefs (Markdown)", data=all_md.encode("utf-8"), file_name="briefs.md", mime="text/markdown")

    # 6) Simple metrics
    st.subheader("Summary")
    st.write({
        "Total questions": int(len(df)),
        "Clusters": int(df["Cluster Label"].nunique()),
        "Avg opportunity": float(df["Opportunity Score"].mean()) if not df.empty else 0.0,
    })

else:
    st.info("Enter a URL and seeds, then click Run Analysis. Use the sidebar toggles to simulate SERP expansion and volume bands.")

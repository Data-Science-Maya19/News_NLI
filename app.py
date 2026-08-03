import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

try:
    import torch
    TORCH_AVAILABLE = True
    TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception as e:
    TORCH_AVAILABLE = False
    TORCH_ERROR = str(e)

import json
import time
import datetime
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Cross-Lingual News NLI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Auto-detect models directory ──────────────────────────────────────────────
def _find_models_dir() -> Path:
    candidates = [
        Path("models"),
        Path("../models"),
        Path(__file__).resolve().parent / "models",
        Path(__file__).resolve().parent.parent / "models",
    ]
    for p in candidates:
        r = p.resolve()
        if r.exists() and r.is_dir():
            if any(r.iterdir()):
                return r
    return Path("models").resolve()

MODELS_DIR = _find_models_dir()

ID2LABEL = {0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"}

NLI_META = {
    "ENTAILMENT": {
        "color": "#166534", "bg": "#dcfce7", "border": "#4ade80",
        "icon": "✓",
        "meaning": "Both articles cover the same event with consistent facts.",
    },
    "NEUTRAL": {
        "color": "#92400e", "bg": "#fef3c7", "border": "#fbbf24",
        "icon": "≈",
        "meaning": "Articles are related but report different aspects.",
    },
    "CONTRADICTION": {
        "color": "#991b1b", "bg": "#fee2e2", "border": "#f87171",
        "icon": "✗",
        "meaning": "Articles present conflicting information or framing.",
    },
}

LANGUAGES = [
    ("Auto-detect", ""),
    ("Arabic (AR)", "ar"),
    ("Chinese (ZH)", "zh"),
    ("English (EN)", "en"),
    ("French (FR)", "fr"),
    ("German (DE)", "de"),
    ("Italian (IT)", "it"),
    ("Polish (PL)", "pl"),
    ("Russian (RU)", "ru"),
    ("Spanish (ES)", "es"),
    ("Turkish (TR)", "tr"),
]
LANG_DISPLAY = [name for name, _ in LANGUAGES]
LANG_CODE    = {name: code for name, code in LANGUAGES}

VALID_PAIRS = {
    "ar-ar", "de-de", "de-en", "de-fr", "de-pl",
    "en-en", "es-en", "es-es", "es-it", "fr-en",
    "fr-fr", "fr-pl", "it-it", "pl-en", "pl-pl",
    "ru-ru", "tr-tr", "zh-en", "zh-zh",
}

BAR_COLORS = {
    "ENTAILMENT":    "#16a34a",
    "NEUTRAL":       "#d97706",
    "CONTRADICTION": "#dc2626",
}

# ═════════════════════════════════════════════════════════════════════════════
# FIX 1 — TEXT-BASED POST-HOC CORRECTION
# Runs at inference time in the app, no annotator scores needed.
# Catches intensity-reversal contradictions ("less" vs "more" etc.)
# and high-overlap pairs where the model is uncertain.
# ═════════════════════════════════════════════════════════════════════════════

# Words that come in opposing pairs — one article says X, other says opposite
INTENSITY_PAIRS = [
    ({"less", "lower", "fewer", "decreased", "reduced", "smaller",
       "weaker", "slower", "lighter", "cheaper", "milder"},
     {"more", "higher", "greater", "increased", "larger", "bigger",
      "stronger", "faster", "heavier", "costlier", "worse", "severe"}),
    ({"rises", "rise", "rising", "grew", "grows", "surges", "climbs",
      "gains", "improves", "recovers", "expands", "boosts"},
     {"falls", "fall", "falling", "shrank", "shrinks", "drops", "slides",
      "loses", "worsens", "declines", "contracts", "cuts"}),
    ({"opens", "launched", "approved", "confirmed", "supports", "backs",
      "agrees", "accepts", "allows", "grants", "wins"},
     {"closes", "cancelled", "rejected", "denied", "opposes", "blocks",
      "disagrees", "refuses", "bans", "loses", "fails"}),
]

# Negation words — if one article has these, stronger contradiction signal
NEGATION_WORDS = {
    "not", "no", "never", "neither", "nor", "without", "deny",
    "denies", "denied", "reject", "rejects", "rejected", "refute",
    "refutes", "contradict", "contradicts",
}

def jaccard_overlap(a: str, b: str) -> float:
    """Word-level Jaccard overlap between two texts."""
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def posthoc_correction(text_a: str, text_b: str, result: dict) -> dict:
    """
    Apply text-based post-hoc correction to catch false ENTAILMENT predictions.

    Three checks (applied in order of confidence):

    1. Intensity reversal — one article uses a "low" word (less, falls, decreases)
       and the other uses the matching "high" word (more, rises, increases).
       If model predicts ENTAILMENT → correct to CONTRADICTION.

    2. Negation flip — one article contains a strong negation word (not, never,
       denied) while sharing high vocabulary overlap with the other.
       If model predicts ENTAILMENT with confidence < 90% → correct to NEUTRAL.

    3. High overlap + low confidence warning — Jaccard overlap >= 0.35 and
       model confidence < 80% on ENTAILMENT. Flag as uncertain (no label change,
       just a warning added to the result).

    Returns the (possibly corrected) result dict with extra fields:
        corrected        : bool  — whether any correction was applied
        correction_reason: str   — human-readable explanation
        original_label   : str   — the raw model label before correction
        jaccard          : float — word overlap between the two texts
    """
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    pred    = result["label"]
    conf    = result["confidence"]
    jac     = jaccard_overlap(text_a, text_b)

    result["original_label"]   = pred
    result["corrected"]        = False
    result["correction_reason"]= None
    result["jaccard"]          = round(jac, 3)
    result["high_overlap_warn"]= False

    # ── Check 1: intensity reversal ───────────────────────────────────────────
    if pred == "ENTAILMENT":
        for group_low, group_high in INTENSITY_PAIRS:
            a_has_low  = bool(words_a & group_low)
            a_has_high = bool(words_a & group_high)
            b_has_low  = bool(words_b & group_low)
            b_has_high = bool(words_b & group_high)

            # One text says "less/falls/…" while the other says "more/rises/…"
            if (a_has_low and b_has_high) or (a_has_high and b_has_low):
                # Find the actual words for a clear explanation
                if a_has_low and b_has_high:
                    word_a = next(iter(words_a & group_low))
                    word_b = next(iter(words_b & group_high))
                else:
                    word_a = next(iter(words_a & group_high))
                    word_b = next(iter(words_b & group_low))

                result["label"]            = "CONTRADICTION"
                result["corrected"]        = True
                result["correction_reason"]= (
                    f"Intensity reversal detected: Article A contains \"{word_a}\" "
                    f"while Article B contains \"{word_b}\" — opposite direction claims."
                )
                # Adjust probabilities to reflect correction
                # Swap ENTAILMENT and CONTRADICTION scores
                result["prob_contradiction"] = result["prob_entailment"]
                result["prob_entailment"]    = result["prob_contradiction"]
                result["confidence"]         = result["prob_contradiction"]
                return result

    # ── Check 2: negation flip (ENTAILMENT + negation + high overlap) ─────────
    if pred == "ENTAILMENT" and conf < 0.90:
        neg_a = bool(words_a & NEGATION_WORDS)
        neg_b = bool(words_b & NEGATION_WORDS)
        if (neg_a or neg_b) and jac >= 0.25:
            neg_word = next(iter((words_a | words_b) & NEGATION_WORDS))
            result["label"]            = "NEUTRAL"
            result["corrected"]        = True
            result["correction_reason"]= (
                f"Negation word \"{neg_word}\" found with high text overlap "
                f"(Jaccard={jac:.2f}) — likely conflicting or qualified claim. "
                f"Downgraded from ENTAILMENT to NEUTRAL."
            )
            return result

    # ── Check 3: high overlap warning (no label change) ───────────────────────
    if pred == "ENTAILMENT" and jac >= 0.35 and conf < 0.85:
        result["high_overlap_warn"] = True

    return result


# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.block-container { padding-top: 1rem !important; }
.ns-title { font-size: 24px; font-weight: 700; color: inherit; letter-spacing: -0.3px; }
.ns-sub   { font-size: 13px; color: #6b7280; margin-bottom: 14px; }
.tag { font-size: 11px; font-weight: 600; padding: 3px 10px;
       border-radius: 99px; display: inline-block; margin-right: 4px; }
.tag-purple { background: #ede9fe; color: #5b21b6; }
.tag-blue   { background: #dbeafe; color: #1d4ed8; }
.pill-ready { display: inline-flex; align-items: center; gap: 5px;
    background: #dcfce7; color: #166534; font-size: 12px; font-weight: 600;
    padding: 4px 12px; border-radius: 99px; }
.dot-green { width: 7px; height: 7px; background: #16a34a;
             border-radius: 50%; display: inline-block; }
.area-label { font-size: 14px; font-weight: 600; color: inherit; margin-bottom: 6px; }
.word-count { font-size: 11px; color: #9ca3af; margin-top: 4px; }
.result-card { border-radius: 14px; padding: 22px 26px; margin-top: 6px; border: 2px solid; }
.result-header { font-size: 11px; font-weight: 600; color: #6b7280;
                 text-transform: uppercase; letter-spacing: 0.5px; }
.result-label  { font-size: 32px; font-weight: 700; margin: 6px 0 4px; }
.result-desc   { font-size: 13px; color: #6b7280; line-height: 1.55; }
.result-meta   { font-size: 11px; color: #9ca3af; margin-top: 10px; }
.correction-badge {
    display: inline-block; font-size: 11px; font-weight: 600;
    padding: 3px 10px; border-radius: 99px; margin-top: 8px;
    background: #fef3c7; color: #92400e; border: 1px solid #fbbf24; }
.prob-title { font-size: 13px; font-weight: 600; color: inherit; margin: 18px 0 10px; }
.prob-row   { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
.prob-name  { font-size: 12px; font-weight: 500; color: inherit; min-width: 120px; }
.prob-track { flex: 1; background: #e5e7eb; border-radius: 99px; height: 9px; overflow: hidden; }
.prob-fill  { height: 9px; border-radius: 99px; }
.prob-pct   { font-size: 12px; font-weight: 700; color: inherit; min-width: 42px; text-align: right; }
.hist-row { border: 1px solid #e5e7eb; border-radius: 10px; padding: 12px 16px;
    margin-bottom: 8px; display: flex; gap: 12px; align-items: flex-start; }
.hist-badge { font-size: 11px; font-weight: 700; padding: 3px 9px;
    border-radius: 99px; white-space: nowrap; flex-shrink: 0; }
.hist-text { font-size: 12px; color: #6b7280; line-height: 1.5; flex: 1; }
.hist-side { font-size: 11px; color: #9ca3af; white-space: nowrap;
    text-align: right; flex-shrink: 0; }
.soft-hr { border: none; border-top: 1px solid #e5e7eb; margin: 16px 0; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_model(model_key: str):
    if not TORCH_AVAILABLE:
        return None, None, None
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    model_dir = MODELS_DIR / model_key
    if not model_dir.exists():
        return None, None, None
    device    = torch.device(TORCH_DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model     = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()
    return tokenizer, model, device


def scan_models() -> list:
    if not MODELS_DIR.exists():
        return []
    models = []
    for folder in sorted(MODELS_DIR.iterdir()):
        if not folder.is_dir():
            continue
        info = {"key": folder.name, "val_f1": None, "hf_name": folder.name}
        sp = folder / "training_summary.json"
        if sp.exists():
            with open(sp) as f:
                s = json.load(f)
            info["val_f1"]  = s.get("best_val_f1_macro")
            info["hf_name"] = s.get("hf_name", folder.name)
            info["lr"]      = s.get("learning_rate")
            info["epochs"]  = s.get("epochs")
        models.append(info)
    return models


# ═════════════════════════════════════════════════════════════════════════════
# PREDICTION  (FIX 1 applied here — post-hoc correction called after model)
# ═════════════════════════════════════════════════════════════════════════════

def run_prediction(text_a: str, text_b: str, model_key: str) -> dict:
    if not TORCH_AVAILABLE:
        return {"error": f"PyTorch failed to load: {TORCH_ERROR}"}

    tokenizer, model, device = load_model(model_key)
    if model is None:
        return {"error": f"Could not load model '{model_key}' from {MODELS_DIR}"}

    enc = tokenizer(
        text_a.strip(), text_b.strip(),
        max_length=128, padding="max_length",
        truncation=True, return_tensors="pt",
    )

    t0 = time.time()
    with torch.no_grad():
        logits = model(
            input_ids      = enc["input_ids"].to(device),
            attention_mask = enc["attention_mask"].to(device),
        ).logits
    latency_ms = (time.time() - t0) * 1000

    probs      = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    pred_id    = int(logits.argmax(dim=-1).cpu())
    pred_label = ID2LABEL[pred_id]

    result = {
        "label":              pred_label,
        "confidence":         float(probs[pred_id]),
        "prob_entailment":    float(probs[0]),
        "prob_neutral":       float(probs[1]),
        "prob_contradiction": float(probs[2]),
        "latency_ms":         round(latency_ms, 1),
        "model_key":          model_key,
    }

    # ── Apply post-hoc correction (Fix 1) ─────────────────────────────────────
    result = posthoc_correction(text_a.strip(), text_b.strip(), result)

    return result


# ═════════════════════════════════════════════════════════════════════════════
# UI — RESULT CARD  (shows correction badge + warning when triggered)
# ═════════════════════════════════════════════════════════════════════════════

def render_result(result: dict):
    if "error" in result:
        st.error(result["error"])
        return

    label = result["label"]
    meta  = NLI_META[label]

    # Build correction note for the card
    correction_html = ""
    if result.get("corrected"):
        orig = result.get("original_label", "")
        correction_html = (
            f'<div class="correction-badge">'
            f'⚙ Auto-corrected from {orig} → {label}'
            f'</div>'
        )

    conf_str     = f"{result['confidence']:.1%}"
    latency_str  = f"{result['latency_ms']:.0f} ms"
    model_str    = result['model_key']
    overlap_str  = f"{result.get('jaccard', 0):.2f}"

    card_html = (
        f'<div class="result-card" style="background:{meta["bg"]}; border-color:{meta["border"]};">'
        f'<div class="result-header">Prediction</div>'
        f'<div class="result-label" style="color:{meta["color"]};">{meta["icon"]}  {label}</div>'
        f'<div class="result-desc">{meta["meaning"]}</div>'
        f'{correction_html}'
        f'<div class="result-meta">'
        f'Confidence <b>{conf_str}</b>'
        f' &nbsp;&middot;&nbsp; {latency_str}'
        f' &nbsp;&middot;&nbsp; model: {model_str}'
        f' &nbsp;&middot;&nbsp; overlap: {overlap_str}'
        f'</div>'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)

    # Show correction explanation as an info box
    if result.get("corrected") and result.get("correction_reason"):
        st.info(f"**Why corrected:** {result['correction_reason']}")

    # High-overlap warning (no label change, just a caution)
    if result.get("high_overlap_warn"):
        st.warning(
            f"⚠ High word overlap (Jaccard = {result.get('jaccard', 0):.2f}) "
            f"with only {result['confidence']:.0%} confidence. "
            "The headlines share many words — verify this prediction manually. "
            "Consider whether one article negates or qualifies the other."
        )

    # Probability bars
    st.markdown('<div class="prob-title">Probability breakdown</div>',
                unsafe_allow_html=True)
    for lbl, key in [("ENTAILMENT",    "prob_entailment"),
                     ("NEUTRAL",       "prob_neutral"),
                     ("CONTRADICTION", "prob_contradiction")]:
        p   = result[key]
        pct = int(p * 100)
        st.markdown(f"""
        <div class="prob-row">
            <span class="prob-name">{lbl}</span>
            <div class="prob-track">
                <div class="prob-fill"
                     style="width:{pct}%; background:{BAR_COLORS[lbl]};"></div>
            </div>
            <span class="prob-pct">{p:.1%}</span>
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGES
# ═════════════════════════════════════════════════════════════════════════════

def page_predict(available_models: list, selected_model):

    if "clear_count" not in st.session_state:
        st.session_state["clear_count"] = 0

    count = st.session_state["clear_count"]
    key_a = f"text_a_{count}"
    key_b = f"text_b_{count}"

    col_a, col_b = st.columns(2, gap="medium")

    with col_a:
        st.markdown("**Article 1** (text_a)")
        lang_a = st.selectbox("Language A", LANG_DISPLAY,
                              key=f"lang_a_{count}",
                              label_visibility="collapsed")
        text_a = st.text_area(
            "text_a", placeholder="Paste first news headline here…",
            height=185, key=key_a, label_visibility="collapsed",
        )
        wc_a = len(text_a.split()) if text_a.strip() else 0
        st.markdown(f'<div class="word-count">{wc_a} words · {LANG_CODE.get(lang_a, "")}</div>',
                    unsafe_allow_html=True)

    with col_b:
        st.markdown("**Article 2** (text_b)")
        lang_b = st.selectbox("Language B", LANG_DISPLAY,
                              key=f"lang_b_{count}",
                              label_visibility="collapsed")
        text_b = st.text_area(
            "text_b", placeholder="Paste second news headline here…",
            height=185, key=key_b, label_visibility="collapsed",
        )
        wc_b = len(text_b.split()) if text_b.strip() else 0
        st.markdown(f'<div class="word-count">{wc_b} words · {LANG_CODE.get(lang_b, "")}</div>',
                    unsafe_allow_html=True)

    # Out-of-distribution language pair warning
    code_a = LANG_CODE.get(lang_a, "")
    code_b = LANG_CODE.get(lang_b, "")
    if code_a and code_b:
        pair     = f"{code_a}-{code_b}"
        pair_rev = f"{code_b}-{code_a}"
        if pair not in VALID_PAIRS and pair_rev not in VALID_PAIRS:
            st.warning(
                f"⚠  The pair **{pair}** was not in the training data. "
                f"Training pairs: {', '.join(sorted(VALID_PAIRS))}"
            )

    st.markdown("<br>", unsafe_allow_html=True)

    b_col, c_col = st.columns([4, 1], gap="small")
    with b_col:
        clicked = st.button("Predict similarity", type="primary",
                            use_container_width=True)
    with c_col:
        if st.button("Clear", use_container_width=True):
            st.session_state["clear_count"] += 1
            st.rerun()

    st.markdown('<hr class="soft-hr">', unsafe_allow_html=True)

    result_slot = st.empty()

    if clicked:
        if not TORCH_AVAILABLE:
            result_slot.error(
                f"**PyTorch failed to load.**\n\nError: `{TORCH_ERROR}`\n\n"
                "Run in terminal:\n```\n"
                "pip uninstall torch torchvision torchaudio -y\n"
                "pip install torch torchvision torchaudio "
                "--index-url https://download.pytorch.org/whl/cu128\n```"
            )
        elif not text_a.strip() and not text_b.strip():
            result_slot.warning("Please enter text in both boxes.")
        elif not text_a.strip():
            result_slot.warning("Please enter text in Article 1.")
        elif not text_b.strip():
            result_slot.warning("Please enter text in Article 2.")
        elif not available_models:
            result_slot.error(
                f"No models found at `{MODELS_DIR}`.\n\n"
                "Train models first using `03_train_per_model_config.ipynb`."
            )
        elif selected_model is None:
            result_slot.warning("Select a model from the dropdown.")
        else:
            with st.spinner(f"Running {selected_model}\u2026"):
                result = run_prediction(
                    text_a.strip(), text_b.strip(), selected_model
                )
            st.session_state["history"].append({
                **result,
                "text_a":    text_a.strip()[:120],
                "text_b":    text_b.strip()[:120],
                "lang_a":    LANG_CODE.get(lang_a, lang_a),
                "lang_b":    LANG_CODE.get(lang_b, lang_b),
                "lang_pair": f"{LANG_CODE.get(lang_a, '')} \u2192 {LANG_CODE.get(lang_b, '')}",
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
            })
            with result_slot.container():
                render_result(result)
    else:
        if st.session_state["history"]:
            with result_slot.container():
                render_result(st.session_state["history"][-1])


def page_history():
    history = st.session_state.get("history", [])
    if not history:
        st.info("No predictions yet.")
        return

    h_col, c_col = st.columns([3, 1])
    with h_col:
        st.markdown(f"**{len(history)} predictions this session**")
    with c_col:
        if st.button("Clear all", use_container_width=True):
            st.session_state["history"] = []
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    for entry in reversed(history):
        if "error" in entry:
            continue
        # Use corrected label if correction was applied
        label = entry.get("label", "NEUTRAL")
        meta  = NLI_META[label]
        conf  = entry.get("confidence", 0)
        corrected_tag = " ⚙" if entry.get("corrected") else ""
        st.markdown(f"""
        <div class="hist-row">
            <span class="hist-badge"
                  style="background:{meta['bg']};color:{meta['color']};
                         border:1px solid {meta['border']};">
                {meta['icon']} {label}{corrected_tag}
            </span>
            <div class="hist-text">
                <b>A:</b> {entry.get('text_a','')[:90]}\u2026<br>
                <b>B:</b> {entry.get('text_b','')[:90]}\u2026
            </div>
            <div class="hist-side">
                {entry.get('timestamp','')}<br>
                {entry.get('lang_pair', f"{entry.get('lang_a','?')} \u2192 {entry.get('lang_b','?')}")}<br>
                <b>{conf:.1%}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)


def page_about(available_models: list):
    st.markdown("### Debug — models path")
    exists = MODELS_DIR.exists()
    found  = [m["key"] for m in available_models]
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Models folder", "Found ✓" if exists else "Missing ✗")
        st.code(str(MODELS_DIR))
    with col2:
        st.metric("Model checkpoints", len(found))
        st.code(str(found) if found else "none")

    st.markdown("---")
    st.markdown("""
### Post-hoc correction (active in this app)

The app automatically corrects two false-entailment patterns that the model commonly gets wrong:

**1. Intensity reversal** — one article says "less / lower / falls" while the other says "more / higher / rises" about the same topic. Example: "variant is *less* severe" vs "variant causes *more* severe illness" → corrected from ENTAILMENT to CONTRADICTION.

**2. Negation flip** — one article contains a negation word (not, denied, rejected) and both articles share high vocabulary overlap. Corrected from ENTAILMENT to NEUTRAL.

**3. High-overlap warning** — Jaccard word overlap ≥ 0.35 with model confidence < 85% on ENTAILMENT → shows a caution message (no label change).

### Labels

| Label | Meaning |
|---|---|
| ENTAILMENT | Same event, consistent facts |
| NEUTRAL | Related but different angles |
| CONTRADICTION | Conflicting information |

### Dataset
SemEval-2022 Task 8 · 18,667 cross-lingual headline pairs · 19 language pair combinations
""")

    if available_models:
        st.markdown("### Trained models")
        import pandas as pd
        rows = []
        for m in available_models:
            sp = MODELS_DIR / m["key"] / "training_summary.json"
            if sp.exists():
                with open(sp) as f:
                    s = json.load(f)
                rows.append({
                    "Model":  m["key"],
                    "Val F1": s.get("best_val_f1_macro", "—"),
                    "LR":     s.get("learning_rate", "—"),
                    "Epochs": s.get("epochs", "—"),
                })
        if rows:
            st.dataframe(pd.DataFrame(rows),
                         use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    available_models = scan_models()

    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "clear_count" not in st.session_state:
        st.session_state["clear_count"] = 0

    left_col, right_col = st.columns([3, 2])
    with left_col:
        st.markdown("## Cross-Lingual News NLI")
        st.caption("Paste two news articles → ENTAILMENT / NEUTRAL / CONTRADICTION")

    with right_col:
        s_col, m_col = st.columns([1, 2])
        with s_col:
            if not TORCH_AVAILABLE:
                st.error("PyTorch error")
            elif available_models:
                st.markdown(
                    '<div style="padding-top:6px">'
                    '<span class="pill-ready">'
                    '<span class="dot-green"></span> Ready'
                    '</span></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.warning("No models")

        with m_col:
            if available_models:
                models_sorted = sorted(
                    available_models,
                    key=lambda m: m["val_f1"] if m["val_f1"] is not None else -1,
                    reverse=True,
                )
                options = {
                    m["key"]: (
                        m["key"].upper()
                        + (f"  ·  F1={m['val_f1']:.3f}" if m["val_f1"] else "  (no F1 saved)")
                    )
                    for m in models_sorted
                }
                selected_model = st.selectbox(
                    "Model", list(options.keys()),
                    format_func=lambda k: options[k],
                    index=0,
                    label_visibility="collapsed",
                )
            else:
                selected_model = None
                st.caption("No trained models found")

    t1, t2, t3 = st.tabs(["Predict", "History", "About"])
    with t1:
        page_predict(available_models, selected_model)
    with t2:
        page_history()
    with t3:
        page_about(available_models)


if __name__ == "__main__":
    main()
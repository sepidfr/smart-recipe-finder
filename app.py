# app.py
# ============================================================
# Smart Recipe Finder (Streamlit)
# - Multiclass cuisine classifier (TF-IDF + Logistic Regression)
# - Recipe generator (structured steps)
# - Lightweight nutrition/tags heuristic
# - Optional English TTS (gTTS) + single-voice “podcast mode”
# - Auto image retrieval via Unsplash source endpoint
# ============================================================

from __future__ import annotations

import io
import json
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Optional, small-footprint TTS
try:
    from gtts import gTTS  # no API key required
    _HAS_TTS = True
except Exception:
    _HAS_TTS = False


# ----------------------------- Paths -----------------------------
APP_DIR = Path(__file__).resolve().parent

# Try several locations so the app works both locally and on Streamlit Cloud
def _locate_file(*candidates: Path) -> Path | None:
    for p in candidates:
        if p and Path(p).exists():
            return Path(p)
    return None

MODEL_PATH = _locate_file(
    APP_DIR / "cuisine_pipeline.joblib",
    APP_DIR / "models" / "cuisine_pipeline.joblib",
    Path("/mnt/data/cuisine_pipeline.joblib"),  # local dev fallback
)

LABELS_PATH = _locate_file(
    APP_DIR / "labels.json",
    APP_DIR / "models" / "labels.json",
    Path("/mnt/data/labels.json"),  # local dev fallback
)

TITLE = "Smart Recipe Finder"
TOPK = 3
np.random.seed(42)


# ------------------------ Loaders (cached) ------------------------
@st.cache_resource(show_spinner="Loading model pipeline...")
def load_pipeline() -> Tuple[object, Dict[int, str]]:
    if MODEL_PATH is None:
        raise FileNotFoundError(
            "Could not find 'cuisine_pipeline.joblib'. Place it in the app root or ./models/."
        )
    pipe = joblib.load(MODEL_PATH)

    if LABELS_PATH is None:
        raise FileNotFoundError(
            "Could not find 'labels.json'. Place it in the app root or ./models/."
        )
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)

    # Accept either a list of class names or a dict {index: name}
    if isinstance(labels, list):
        inv = {i: name for i, name in enumerate(labels)}
    elif isinstance(labels, dict):
        inv = {int(k): v for k, v in labels.items()}
    else:
        raise TypeError("labels.json must be a list of class names or a dict {index: name}.")

    return pipe, inv


# --------------------------- Utilities ---------------------------
def cuisine_image_url(cuisine_name: str) -> str:
    # License-friendly source (no API key). Loads client-side.
    query = f"{cuisine_name} dish plated"
    return f"https://source.unsplash.com/800x500/?{query.replace(' ', '%20')}"


def normalize_ingredients_block(text: str) -> List[str]:
    raw = [t.strip() for t in text.replace("\n", ",").split(",")]
    return [t for t in raw if t]


def predict_topk(pipe, inv_labels: Dict[int, str], ingredients: List[str], k: int = TOPK):
    text = " ".join(ingredients)
    probs = pipe.predict_proba([text])[0]
    order = np.argsort(probs)[::-1][:k]
    names = [inv_labels[i] for i in order]
    values = probs[order]
    return names, values, probs


def qualitative_nutrition(ings: List[str]) -> Dict[str, float | str]:
    s = " ".join(ings).lower()

    hi_cal = sum(w in s for w in
                 ["butter", "oil", "olive oil", "ghee", "cream", "cheese", "sugar", "fried", "bacon", "nuts", "peanut"])
    protein = sum(w in s for w in
                  ["chicken", "beef", "pork", "fish", "tuna", "egg", "tofu", "lentil", "bean", "chickpea", "yogurt", "paneer"])
    greens = sum(w in s for w in
                 ["spinach", "kale", "broccoli", "herb", "parsley", "cilantro", "tomato", "cucumber", "lettuce", "carrot", "zucchini", "pepper"])

    c_score = min(1.0, 0.20 * hi_cal + 0.15 * len(ings))
    p_score = min(1.0, 0.12 * protein + 0.02 * len(ings))
    h_score = min(1.0, 0.10 * greens + 0.02 * (len(ings) - hi_cal))

    band = "low"
    if c_score > 0.66:
        band = "high"
    elif c_score > 0.33:
        band = "medium"

    return {
        "caloric_density": float(c_score),
        "protein_index": float(p_score),
        "healthiness": float(h_score),
        "calorie_band": band,
    }


def dietary_tags(ings: List[str]) -> List[str]:
    s = " ".join(ings).lower()
    tags = []
    animal = any(w in s for w in ["chicken", "beef", "pork", "fish", "shrimp", "egg", "yogurt", "cheese", "milk", "butter", "honey"])
    dairy = any(w in s for w in ["cheese", "milk", "yogurt", "cream", "butter", "ghee"])
    gluten = any(w in s for w in ["flour", "bread", "pasta", "noodle", "wheat", "semolina", "bulgur", "couscous"])
    pork = "pork" in s or "bacon" in s or "ham" in s
    alcohol = any(w in s for w in ["wine", "beer", "ale", "vodka", "rum", "whiskey", "brandy"])

    if not animal and not dairy:
        tags.append("vegan-ish")
    elif not animal:
        tags.append("vegetarian-ish")

    if not gluten:
        tags.append("gluten-light")
    if not pork:
        tags.append("pork-free")
    if not alcohol:
        tags.append("no-alcohol")
    if any(w in s for w in ["chili", "chilli", "jalapeno", "cayenne", "gochujang", "harissa", "pepper flakes"]):
        tags.append("spicy")

    return tags[:4]


def generate_recipe(cuisine: str, ings: List[str]) -> str:
    title = f"{cuisine.title()} Style Dish with {', '.join(ings[:3]).title() if ings else 'Seasonal Ingredients'}"
    steps = [
        "Prep: finely chop aromatics (garlic/ginger/onion) and measure spices.",
        "Season main ingredients with salt and pepper.",
        "Heat oil in a pan; bloom spices and aromatics until fragrant.",
        "Add main ingredients, sear lightly, then cook through.",
        "Adjust with acid (lemon/lime/vinegar) and fresh herbs.",
        "Taste, correct seasoning, and serve warm.",
    ]
    flair = {
        "indian": "Finish with garam masala and fresh cilantro.",
        "chinese": "Balance with a 1–1 soy sauce and rice vinegar splash; add sesame oil off heat.",
        "italian": "Deglaze with a touch of white wine; finish with olive oil and basil.",
        "mexican": "Add cumin and chili powder; finish with lime and cilantro.",
        "japanese": "Season with a dash of mirin and soy; garnish with scallion.",
        "korean": "Stir in gochujang for heat and depth; top with sesame seeds.",
        "french": "Mount with a small knob of butter; add parsley and chives.",
        "thai": "Balance sweet–sour–salty with palm sugar, lime, and fish sauce.",
    }
    steps.append(flair.get(cuisine.lower(), "Finish with fresh herbs and a drizzle of olive oil."))

    return f"{title}\n\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))


def tts_bytes_en(text: str) -> bytes | None:
    if not _HAS_TTS:
        return None
    tts = gTTS(text=text, lang="en")
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    return buf.getvalue()


def build_podcast_script(cuisine: str, ings: List[str], nutr: Dict[str, float | str], tags: List[str]) -> str:
    ttags = ", ".join(tags) if tags else "balanced"
    return textwrap.dedent(f"""
    HOST: Welcome to Quick Plates! Today we're exploring {cuisine.title()} cuisine.
    HOST: Our basket ingredients are: {', '.join(ings)}.

    CHEF: Great pick! This style balances tradition with bold flavors.
    CHEF: For a quick home version, sauté aromatics, add your main ingredient, and finish with regional staples.

    HOST: Nutrition snapshot?
    CHEF: Caloric density looks {nutr['calorie_band']}, protein index {nutr['protein_index']:.2f}, overall healthiness {nutr['healthiness']:.2f}.
    CHEF: Dietary hints: {ttags}.

    HOST: Any final flourish?
    CHEF: Always taste and balance at the end. Fresh herbs and acid bring the dish alive.

    HOST: Thanks for cooking with us!
    """).strip()


# ------------------------------ UI ------------------------------
st.set_page_config(page_title=TITLE, page_icon="🍽️", layout="wide")
st.title(TITLE)
st.caption("TF-IDF + Logistic Regression • Multiclass cuisine classifier + recipe generator")

# Sidebar
st.sidebar.header("Settings")
enable_tts = st.sidebar.checkbox("Enable voice (English TTS)", value=False, disabled=not _HAS_TTS)
enable_podcast = st.sidebar.checkbox("Podcast mode (single-voice)", value=False, disabled=not _HAS_TTS)

with st.sidebar.expander("About the model", expanded=True):
    st.markdown("- Logistic Regression + TF-IDF (multiclass)")

pipe, INV = load_pipeline()
st.sidebar.markdown(f"- Classes: **{len(INV)}**")

# Layout
col_left, col_right = st.columns([1.25, 1.0], vertical_alignment="top")

with col_left:
    st.subheader("Ingredients")
    demo = "chicken, soy sauce, ginger, garlic, sesame oil"
    ing_text = st.text_area(
        "Comma-separated or one per line",
        value=demo,
        height=120,
        placeholder="e.g., tomato, basil, garlic, olive oil",
    )
    ings = normalize_ingredients_block(ing_text)

    if st.button("Find cuisine & build recipe", type="primary", use_container_width=True):
        if not ings:
            st.warning("Please provide at least one ingredient.")
        else:
            names, values, all_probs = predict_topk(pipe, INV, ings, k=TOPK)

            st.subheader("Top predictions")
            df_plot = pd.DataFrame({"cuisine": names, "probability": values})
            st.bar_chart(df_plot, x="cuisine", y="probability", use_container_width=True)

            top1 = names[0]
            st.markdown(f"### Image • {top1.title()}")
            st.image(cuisine_image_url(top1), use_column_width=True)

            st.markdown("### 🧾 Generated Recipe")
            recipe_text = generate_recipe(top1, ings)
            st.text_area("Recipe", value=recipe_text, height=230, label_visibility="collapsed")

            nutr = qualitative_nutrition(ings)
            tags = dietary_tags(ings)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Caloric density (0–1)", f"{nutr['caloric_density']:.2f}")
            with c2:
                st.metric("Protein index (0–1)", f"{nutr['protein_index']:.2f}")
            with c3:
                st.metric("Healthiness (0–1)", f"{nutr['healthiness']:.2f}")
            st.caption(f"Dietary tags: {', '.join(tags) if tags else '—'}")

            if enable_tts and _HAS_TTS:
                st.markdown("#### 🔊 Read recipe (English)")
                audio_bytes = tts_bytes_en(recipe_text)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3")

            if enable_podcast and _HAS_TTS:
                st.markdown("#### 🎙️ Auto-podcast")
                script = build_podcast_script(top1, ings, nutr, tags)
                st.text_area("Podcast transcript", value=script, height=180, label_visibility="collapsed")
                audio_pod = tts_bytes_en(script)
                if audio_pod:
                    st.audio(audio_pod, format="audio/mp3")

with col_right:
    st.subheader("How to use")
    st.markdown(
        """
        1. Enter ingredients (comma-separated or one per line).  
        2. Click **Find cuisine & build recipe**.  
        3. Review predictions, image, recipe, and nutrition hints.  
        4. (Optional) Enable **voice** or **podcast** in the sidebar.
        """
    )

st.markdown("---")
st.caption(
    "Business demo ideas: meal helper, shopping-list recommender, cooking coach. "
    "Place **cuisine_pipeline.joblib** and **labels.json** in the repo root (or ./models/)."
)

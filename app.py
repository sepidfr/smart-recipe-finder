# app.py
# ============================================================
# Smart Recipe Finder • Cuisine Classification + Recipe Builder
# Model: TF–IDF + Logistic Regression (saved as joblib)
# UI: Streamlit (top-k predictions, image, recipe text, nutrition,
#      Food Value Score, optional English TTS and podcast mode)
# ------------------------------------------------------------
# Files expected in the app root:
#   - cuisine_pipeline.joblib
#   - labels.json   (list of class names OR {index->name} dict)
# ============================================================

from __future__ import annotations
import io, json, re, textwrap
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ---------- TTS (optional) ----------
def _safe_import_gtts():
    try:
        from gtts import gTTS
        return gTTS
    except Exception:
        return None
gTTS = _safe_import_gtts()

# ---------- Paths ----------
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "cuisine_pipeline.joblib"
LABELS_PATH = APP_DIR / "labels.json"

# ---------- Constants ----------
TITLE = "Smart Recipe Finder"
TOPK = 3
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ---------- Nutrition table (per 100 g) ----------
NUTR_TABLE = {
    "chicken":      {"kcal":165, "protein":31.0, "fat":3.6,  "carbs":0.0},
    "beef":         {"kcal":217, "protein":26.1, "fat":11.8, "carbs":0.0},
    "pork":         {"kcal":242, "protein":27.0, "fat":14.0, "carbs":0.0},
    "tofu":         {"kcal": 76, "protein": 8.0, "fat":4.8,  "carbs":1.9},
    "lentil":       {"kcal":116, "protein": 9.0, "fat":0.4,  "carbs":20.1},
    "bean":         {"kcal":127, "protein": 8.7, "fat":0.5,  "carbs":22.8},
    "chickpea":     {"kcal":164, "protein": 8.9, "fat":2.6,  "carbs":27.4},
    "garlic":       {"kcal":149, "protein": 6.4, "fat":0.5,  "carbs":33.1},
    "ginger":       {"kcal": 80, "protein": 1.8, "fat":0.8,  "carbs":17.8},
    "onion":        {"kcal": 40, "protein": 1.1, "fat":0.1,  "carbs":9.3},
    "tomato":       {"kcal": 18, "protein": 0.9, "fat":0.2,  "carbs":3.9},
    "basil":        {"kcal": 23, "protein": 3.2, "fat":0.6,  "carbs":2.7},
    "olive oil":    {"kcal":884, "protein": 0.0,"fat":100.0,"carbs":0.0},
    "sesame oil":   {"kcal":884, "protein": 0.0,"fat":100.0,"carbs":0.0},
    "butter":       {"kcal":717, "protein": 0.9,"fat":81.1, "carbs":0.1},
    "soy sauce":    {"kcal": 53, "protein": 8.0,"fat":0.6,  "carbs":5.6},
    "rice":         {"kcal":130, "protein": 2.4,"fat":0.3,  "carbs":28.0},
    "pasta":        {"kcal":131, "protein": 5.0,"fat":1.1,  "carbs":25.0},
    "yogurt":       {"kcal": 59, "protein":10.0,"fat":0.4,  "carbs":3.6},
}

# ---------- Caching ----------
@st.cache_resource(show_spinner="Loading model pipeline...")
def load_pipeline() -> Tuple[object, Dict[int, str]]:
    if not MODEL_PATH.exists():
        st.error(f"Model not found at {MODEL_PATH}. Upload cuisine_pipeline.joblib to the app root.")
        st.stop()
    pipe = joblib.load(MODEL_PATH)

    if not LABELS_PATH.exists():
        st.error(f"Labels file not found at {LABELS_PATH}. Upload labels.json to the app root.")
        st.stop()
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)

    # Support BOTH formats: list or dict
    if isinstance(labels, list):
        inv = {i: name for i, name in enumerate(labels)}
    elif isinstance(labels, dict):
        inv = {int(k): v for k, v in labels.items()}
    else:
        st.error("labels.json must be a list of names or a {index->name} dict.")
        st.stop()

    return pipe, inv

# ---------- Helpers ----------
def cuisine_image_url(cuisine_name: str) -> str:
    query = f"{cuisine_name} plated dish"
    return f"https://source.unsplash.com/800x500/?{query.replace(' ', '%20')}"

def parse_ingredients(text: str) -> List[str]:
    raw = [t.strip() for t in text.replace("\n", ",").split(",")]
    return [t for t in raw if t]

def parse_mass_g(s: str, default: float = 100.0) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(g|gram|grams)\b", s.lower())
    return float(m.group(1)) if m else float(default)

def estimate_nutrition(items: List[str], default_mass_g: float = 100.0) -> Dict[str, float]:
    tot = {"kcal":0.0, "protein":0.0, "fat":0.0, "carbs":0.0}
    for it in items:
        it_l = it.lower()
        grams = parse_mass_g(it_l, default_mass_g)
        for key, nt in NUTR_TABLE.items():
            if key in it_l:
                mult = grams / 100.0
                tot["kcal"]    += nt["kcal"]    * mult
                tot["protein"] += nt["protein"] * mult
                tot["fat"]     += nt["fat"]     * mult
                tot["carbs"]   += nt["carbs"]   * mult
                break
    return tot

def predict_topk(pipe, inv_labels: Dict[int, str], ingredients: List[str], k: int = TOPK):
    text = " ".join(ingredients)
    proba = pipe.predict_proba([text])[0]
    order = np.argsort(proba)[::-1][:k]
    names = [inv_labels[i] for i in order]
    values = proba[order]
    return names, values, proba

def qualitative_nutrition(ings: List[str]) -> Dict[str, float | str]:
    s = " ".join(ings).lower()
    hi_cal = sum(w in s for w in [
        "butter","oil","olive oil","ghee","cream","cheese","sugar","fried","bacon","nuts","peanut","sesame oil"
    ])
    protein = sum(w in s for w in [
        "chicken","beef","pork","fish","tuna","egg","tofu","lentil","bean","chickpea","yogurt","paneer"
    ])
    greens = sum(w in s for w in [
        "spinach","kale","broccoli","herb","parsley","cilantro","tomato","cucumber","lettuce","carrot","zucchini","pepper"
    ])

    c_score = min(1.0, 0.20*hi_cal + 0.15*len(ings))
    p_score = min(1.0, 0.12*protein + 0.02*len(ings))
    h_score = min(1.0, 0.10*greens + 0.02*(len(ings) - hi_cal))

    cal_band = "low" if c_score <= 0.33 else ("medium" if c_score <= 0.66 else "high")
    return {
        "caloric_density": float(c_score),
        "protein_index": float(p_score),
        "healthiness": float(h_score),
        "calorie_band": cal_band,
    }

def dietary_tags(ings: List[str]) -> List[str]:
    s = " ".join(ings).lower()
    tags = []
    animal = any(w in s for w in ["chicken","beef","pork","fish","shrimp","egg","yogurt","cheese","milk","butter","honey"])
    dairy  = any(w in s for w in ["cheese","milk","yogurt","cream","butter","ghee"])
    gluten = any(w in s for w in ["flour","bread","pasta","noodle","wheat","semolina","bulgur","couscous"])
    pork   = ("pork" in s) or ("bacon" in s) or ("ham" in s)
    alcohol= any(w in s for w in ["wine","beer","ale","vodka","rum","whiskey","brandy"])

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
    if any(w in s for w in ["chili","chilli","jalapeno","cayenne","gochujang","harissa","pepper flakes"]):
        tags.append("spicy")
    return tags[:4]

def food_value_score(nutr: Dict[str, float]) -> float:
    score = 0.35*nutr["protein_index"] + 0.40*nutr["healthiness"] - 0.25*nutr["caloric_density"]
    return float(np.clip(score, 0.0, 1.0))

def generate_recipe(cuisine: str, ings: List[str]) -> str:
    title = f"{cuisine.title()} Style Dish with {', '.join(ings[:3]).title() if ings else 'Seasonal Ingredients'}"
    steps = [
        "Finely chop aromatics (garlic/ginger/onion) and measure spices.",
        "Season main ingredients with salt and pepper.",
        "Heat oil; bloom spices and aromatics until fragrant.",
        "Add main ingredients, sear lightly, then cook through.",
        "Adjust with acid (lemon/lime/vinegar) and fresh herbs.",
        "Taste, correct seasoning, and serve warm."
    ]
    flair = {
        "indian":  "Finish with garam masala and fresh cilantro.",
        "chinese": "Add a 1–1 splash of soy sauce and rice vinegar; sesame oil off-heat.",
        "italian": "Deglaze with a touch of white wine; finish with olive oil and basil.",
        "mexican": "Add cumin and chili powder; finish with lime and cilantro.",
        "japanese":"Season with mirin and soy; garnish with scallion.",
        "korean":  "Stir in gochujang; top with sesame seeds.",
        "french":  "Mount with a small knob of butter; finish with parsley/chives.",
        "thai":    "Balance sweet–sour–salty with palm sugar, lime, fish sauce."
    }
    steps.append(flair.get(cuisine.lower(), "Finish with fresh herbs and a drizzle of good olive oil."))
    return f"{title}\n\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))

def tts_bytes_en(text: str) -> bytes | None:
    if gTTS is None:
        return None
    try:
        tts = gTTS(text=text, lang="en")
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        return buf.getvalue()
    except Exception:
        return None

def build_podcast_script(cuisine: str, ings: List[str], nutr: Dict[str, float], tags: List[str]) -> str:
    ttags = ", ".join(tags) if tags else "balanced"
    return textwrap.dedent(f"""
    HOST: Welcome to Quick Plates. Today we’re exploring {cuisine.title()} flavors.
    HOST: Our basket has {', '.join(ings)}.

    CHEF: Great pick. For a fast home version, sauté aromatics, add your main ingredient, and finish with regional staples.

    HOST: Quick nutrition?
    CHEF: Caloric density is {nutr['calorie_band']}, protein index {nutr['protein_index']:.2f}, healthiness {nutr['healthiness']:.2f}.
    CHEF: Dietary hints: {ttags}.

    HOST: Final touch?
    CHEF: Always taste and balance at the end—acid and fresh herbs bring the dish to life.
    """).strip()

# ======================= UI =======================
st.set_page_config(page_title=TITLE, page_icon="🍽️", layout="wide")
st.title(TITLE)
st.caption("Cuisine prediction + auto recipe + nutrition hints")

# Sidebar
st.sidebar.header("Settings")
enable_tts = st.sidebar.checkbox("Enable voice for recipe (English TTS)", value=False)
enable_podcast = st.sidebar.checkbox("Podcast script + voice (English)", value=False)

with st.sidebar.expander("About the model", expanded=True):
    st.markdown("- Logistic Regression over TF–IDF features\n- Trained on the Yummly ‘What’s Cooking?’ dataset")

# Load model/labels
pipe, INV = load_pipeline()
st.sidebar.markdown(f"- Classes: **{len(INV)}**")

# Layout
left, right = st.columns([1.25, 1.0], vertical_alignment="top")

with left:
    st.subheader("Ingredients")
    demo = "chicken, soy sauce, ginger, garlic, sesame oil"
    ing_text = st.text_area("Comma-separated or one per line", value=demo, height=120,
                            placeholder="e.g., tomato, basil, garlic, olive oil")
    ings = parse_ingredients(ing_text)

    run = st.button("Find cuisine & build recipe", type="primary", use_container_width=True)
    if run:
        if not ings:
            st.warning("Please provide at least one ingredient.")
            st.stop()

        names, values, all_probs = predict_topk(pipe, INV, ings, k=TOPK)

        st.markdown("### Prediction & Recipe")

        # --- Top cuisines (left) + Nutrition (right)
        col1, col2 = st.columns([3, 2], vertical_alignment="top")

        with col1:
            st.markdown("#### Top cuisines")
            df_pred = pd.DataFrame({"cuisine": names, "probability": values})
            st.bar_chart(df_pred.set_index("cuisine")["probability"], use_container_width=True)

        with col2:
            st.markdown("#### Nutrition (approx. per serving)")
            # numeric macro estimate from ingredient list
            macro = estimate_nutrition(ings, default_mass_g=100.0)
            nutr = qualitative_nutrition(ings)
            tags = dietary_tags(ings)
            score = food_value_score(nutr)

            st.metric("Calories", f"{macro['kcal']:.0f} kcal")
            a, b, c = st.columns(3)
            a.metric("Protein", f"{macro['protein']:.1f} g")
            b.metric("Fat",     f"{macro['fat']:.1f} g")
            c.metric("Carbs",   f"{macro['carbs']:.1f} g")

            st.caption(f"Caloric density (0–1): {nutr['caloric_density']:.2f} • "
                       f"Protein index (0–1): {nutr['protein_index']:.2f} • "
                       f"Healthiness (0–1): {nutr['healthiness']:.2f} • "
                       f"Food Value Score: {score:.2f}")
            st.caption(f"Dietary tags: {', '.join(tags) if tags else '—'}")

        # --- Image + Recipe
        top1 = names[0]
        st.markdown(f"### Image • {top1.title()}")
        st.image(cuisine_image_url(top1), use_column_width=True)

        st.markdown("### 🧾 Generated Recipe")
        recipe_text = generate_recipe(top1, ings)
        st.text_area("Recipe", value=recipe_text, height=220, label_visibility="collapsed")

        # --- Optional audio
        if enable_tts:
            st.markdown("#### 🔊 Read recipe (English)")
            audio = tts_bytes_en(recipe_text)
            if audio:
                st.audio(audio, format="audio/mp3")
            else:
                st.info("TTS unavailable in this environment.")

        if enable_podcast:
            st.markdown("#### 🎙️ Podcast")
            script = build_podcast_script(top1, ings, qualitative_nutrition(ings), dietary_tags(ings))
            st.text_area("Podcast transcript", value=script, height=180, label_visibility="collapsed")
            audio2 = tts_bytes_en(script)
            if audio2:
                st.audio(audio2, format="audio/mp3")
            else:
                st.info("TTS unavailable in this environment.")

with right:
    st.subheader("How to use")
    st.markdown(
        "1) Enter ingredients\n\n"
        "2) Click **Find cuisine & build recipe**\n\n"
        "3) Review predictions (left) and **Nutrition** (right)\n\n"
        "4) See image, recipe, and optionally enable **voice** or **podcast**"
    )

st.markdown("---")
st.caption("Demo idea: meal helper / shopping assistant / cooking coach.")

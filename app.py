# app.py
# ============================================================
# Smart Recipe Finder
# - TF–IDF + Logistic Regression pipeline (joblib)
# - Top-k cuisine prediction, image, generated recipe
# - Heuristic nutrition + Food Value Score (0–1)
# - Optional English TTS (gTTS) + natural single-voice podcast
# ============================================================

from __future__ import annotations
import io
import json
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib
import streamlit as st

from gtts import gTTS  # simple text-to-speech (English)

# ---------- Paths ----------
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "cuisine_pipeline.joblib"
LABELS_PATH = APP_DIR / "labels.json"

# ---------- Constants ----------
TITLE = "Smart Recipe Finder"
TOPK = 3
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ---------- Cache loaders ----------
@st.cache_resource(show_spinner="Loading model pipeline...")
def load_pipeline() -> Tuple[object, Dict[int, str]]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Place cuisine_pipeline.joblib in the app folder."
        )
    pipe = joblib.load(MODEL_PATH)

    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Labels file not found at {LABELS_PATH}. Place labels.json (int->str mapping) in the app folder."
        )
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)

    # Ensure keys are ints
    inv = {int(k): v for k, v in labels.items()}
    return pipe, inv


# ---------- Helpers ----------
def cuisine_image_url(cuisine_name: str) -> str:
    # License-friendly Unsplash source endpoint (no API key)
    q = f"{cuisine_name} dish plated"
    return f"https://source.unsplash.com/800x500/?{q.replace(' ', '%20')}"

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

def qualitative_nutrition(ings: List[str]) -> Dict[str, float]:
    """
    Heuristic proxies in [0,1]:
      - caloric_density  ↑ with fats/sugars and length
      - protein_index    ↑ with lean proteins/legumes
      - healthiness      ↑ with greens/herbs/vegetables
    """
    s = " ".join(ings).lower()

    hi_cal = sum(w in s for w in [
        "butter","oil","olive oil","ghee","cream","cheese","sugar","fried","bacon","nuts","peanut"
    ])
    protein = sum(w in s for w in [
        "chicken","beef","pork","fish","tuna","egg","tofu","lentil","bean","chickpea","yogurt","paneer"
    ])
    greens = sum(w in s for w in [
        "spinach","kale","broccoli","herb","parsley","cilantro","tomato","cucumber","lettuce","carrot","zucchini","pepper"
    ])

    c_score = min(1.0, 0.20*hi_cal + 0.15*len(ings))
    p_score = min(1.0, 0.12*protein + 0.02*len(ings))
    h_score = min(1.0, 0.10*greens + 0.02*max(0, len(ings)-hi_cal))

    cal_band = "low"
    if c_score > 0.66:
        cal_band = "high"
    elif c_score > 0.33:
        cal_band = "medium"

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
    """
    Aggregate 'value' of the dish in [0,1],
    prioritizing healthiness & protein, penalizing high caloric density.
    """
    c = nutr["caloric_density"]
    p = nutr["protein_index"]
    h = nutr["healthiness"]
    score = 0.4*h + 0.4*p + 0.2*(1.0 - c)
    return float(max(0.0, min(1.0, score)))

def generate_recipe(cuisine: str, ings: List[str]) -> str:
    title = f"{cuisine.title()} Style Dish with {', '.join(ings[:3]).title() if ings else 'Seasonal Ingredients'}"
    steps = [
        "Prep: finely chop aromatics and measure spices.",
        "Season main ingredients with salt and pepper.",
        "Heat oil; bloom spices and aromatics until fragrant.",
        "Add main ingredients, sear lightly, then cook through.",
        "Adjust with acid (lemon/lime/vinegar) and fresh herbs.",
        "Taste, correct seasoning, and serve warm."
    ]
    flourish = {
        "indian": "Finish with garam masala and fresh cilantro.",
        "chinese": "Balance soy and rice vinegar; add sesame oil off-heat.",
        "italian": "Deglaze with a touch of white wine; finish with olive oil and basil.",
        "mexican": "Cumin + chili powder; finish with lime and cilantro.",
        "japanese": "Dash of mirin and soy; garnish with scallion.",
        "korean": "Stir in gochujang; top with sesame seeds.",
        "french": "Mount with a small knob of butter; add parsley and chives.",
        "thai": "Balance sweet/sour/salty with palm sugar, lime, and fish sauce."
    }
    steps.append(flourish.get(cuisine.lower(), "Finish with fresh herbs and a drizzle of good olive oil."))
    return f"{title}\n\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))

def tts_bytes_en(text: str) -> bytes:
    tts = gTTS(text=text, lang="en")
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    return buf.getvalue()

def build_podcast_script(cuisine: str, ings: List[str], nutr: Dict[str, float], tags: List[str]) -> str:
    """
    Natural one-voice, dialogue-style script (no HOST/CHEF labels).
    """
    tag_string = ", ".join(tags) if tags else "balanced"
    return textwrap.dedent(f"""
    - Today we're exploring {cuisine.title()} cooking. On the counter we’ve got {', '.join(ings)}.
    - Nice selection. This style blends comfort with bold, clean flavors.

    - If we were making a quick version at home, where would we start?
    - I’d warm some oil, open up the aromatics, then add the main ingredient and let the spices bloom.

    - And nutritionally?
    - Caloric density looks {nutr['calorie_band']}; protein index is around {nutr['protein_index']:.2f}, healthiness near {nutr['healthiness']:.2f}.
      Dietary note: {tag_string}.

    - Any final flourish?
    - A bright finish—fresh herbs and a squeeze of citrus—always lifts the dish.

    - Perfect. That’s our quick kitchen chat for today.
    """).strip()


# ------------------------ UI ------------------------
st.set_page_config(page_title=TITLE, page_icon="🍽️", layout="wide")
st.title(TITLE)
st.caption("Multiclass cuisine classifier + recipe generator (TF–IDF + Logistic Regression)")

# Sidebar
st.sidebar.header("Settings")
enable_tts = st.sidebar.checkbox("Enable voice (English TTS)", value=False)
enable_podcast = st.sidebar.checkbox("Podcast mode (natural single voice)", value=False)

with st.sidebar.expander("About the model", expanded=True):
    st.markdown("- Logistic Regression + TF-IDF (multiclass)")
pipe, INV = load_pipeline()
st.sidebar.markdown(f"- Classes: {len(INV)}")

# Main layout
col_left, col_right = st.columns([1.25, 1.0], vertical_alignment="top")

with col_left:
    st.subheader("Ingredients")
    default_demo = "chicken, soy sauce, ginger, garlic, sesame oil"
    ing_text = st.text_area(
        "Comma-separated or one per line",
        value=default_demo,
        height=120,
        placeholder="e.g., tomato, basil, garlic, olive oil",
    )
    ings = normalize_ingredients_block(ing_text)

    if st.button("Find cuisine & build recipe", type="primary", use_container_width=True):
        if not ings:
            st.warning("Please provide at least one ingredient.")
        else:
            names, values, _ = predict_topk(pipe, INV, ings, k=TOPK)

            # --- Predictions chart ---
            st.subheader("Top predictions")
            df_plot = pd.DataFrame({"cuisine": names, "probability": values})
            st.bar_chart(df_plot, x="cuisine", y="probability", use_container_width=True)

            # --- Image ---
            top1 = names[0]
            st.markdown(f"### Image • {top1.title()}")
            st.image(cuisine_image_url(top1), use_column_width=True)

            # --- Recipe ---
            st.markdown("### 🧾 Generated Recipe")
            recipe_text = generate_recipe(top1, ings)
            st.text_area("Recipe", value=recipe_text, height=220, label_visibility="collapsed")

            # --- Nutrition + Tags ---
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

            # --- Food Value Score ---
            st.markdown("### 🍽️ Food Value Score")
            score = food_value_score(nutr)
            st.progress(score)
            st.write(f"**Score:** {score:.2f}  — combines healthiness + protein and penalizes excess caloric density.")

            # --- TTS (Recipe) ---
            if enable_tts:
                st.markdown("#### 🔊 Read recipe (English)")
                try:
                    st.audio(tts_bytes_en(recipe_text), format="audio/mp3")
                except Exception as e:
                    st.info(f"TTS unavailable: {e}")

            # --- Podcast Mode ---
            if enable_podcast:
                st.markdown("#### 🎙️ Podcast")
                script = build_podcast_script(top1, ings, nutr, tags)
                st.text_area("Podcast transcript", value=script, height=180, label_visibility="collapsed")
                try:
                    st.audio(tts_bytes_en(script), format="audio/mp3")
                except Exception as e:
                    st.info(f"TTS unavailable: {e}")

with col_right:
    st.subheader("How to use")
    st.markdown(
        """
        1) Enter ingredients (comma-separated or one per line).  
        2) Click **Find cuisine & build recipe**.  
        3) Review predictions, image, recipe, nutrition, and Food Value Score.  
        4) (Optional) Enable **voice** or **podcast** in the sidebar.
        """
    )

st.markdown("---")
st.caption("Tip: Package this as a meal helper / shopping recommender / cooking coach.")

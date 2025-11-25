# app.py
# ============================================================
# Recipe Finder • Cuisine Classification + Recipe Generator
# TF–IDF + Logistic Regression (pipeline saved as joblib)
# Streamlit UI with: chart, image, recipe text, nutrition tags,
# and optional English TTS (gTTS) including a "podcast mode".
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

from gtts import gTTS  # simple, no-API text-to-speech

# ---------- Paths ----------
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "cuisine_pipeline.joblib"
LABELS_PATH = APP_DIR / "labels.json"

# ---------- Constants ----------
TITLE = "Smart Recipe Finder"
TOPK = 3  # show top-k cuisines
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ---------- Cache loaders ----------
@st.cache_resource(show_spinner="Loading model pipeline...")
def load_pipeline() -> Tuple[object, Dict[int, str]]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Upload cuisine_pipeline.joblib to the app root."
        )
    pipe = joblib.load(MODEL_PATH)

    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Labels file not found at {LABELS_PATH}. Upload labels.json (int->str mapping)."
        )

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)

    # Ensure keys are ints
    inv = {int(k): v for k, v in labels.items()}
    return pipe, inv


# ---------- Simple helpers ----------
def cuisine_image_url(cuisine_name: str) -> str:
    """
    No external API: use Unsplash's source endpoint to fetch a representative, license-friendly photo.
    This URL can be passed directly to st.image and loads on the client.
    """
    query = f"{cuisine_name} dish plated"
    return f"https://source.unsplash.com/800x500/?{query.replace(' ', '%20')}"


def pretty_ingredients_box(text: str) -> List[str]:
    # Split by comma or newline; strip/normalize
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
    Lightweight nutrition estimate: score [0..1] proxies for caloric density, protein, and 'healthiness'.
    Heuristic: fats/sugars ↑ calories; lean proteins ↑ protein; greens/herbs ↑ healthiness.
    """
    s = " ".join(ings).lower()

    hi_cal = sum(w in s for w in ["butter", "oil", "olive oil", "ghee", "cream", "cheese", "sugar", "fried", "bacon", "nuts", "peanut"])
    protein = sum(w in s for w in ["chicken", "beef", "pork", "fish", "tuna", "egg", "tofu", "lentil", "bean", "chickpea", "yogurt", "paneer"])
    greens = sum(w in s for w in ["spinach", "kale", "broccoli", "herb", "parsley", "cilantro", "tomato", "cucumber", "lettuce", "carrot", "zucchini", "pepper"])

    # Normalize to [0,1]
    c_score = min(1.0, 0.20 * hi_cal + 0.15 * len(ings))
    p_score = min(1.0, 0.12 * protein + 0.02 * len(ings))
    h_score = min(1.0, 0.10 * greens + 0.02 * (len(ings) - hi_cal))

    # "Calories per serving" rough category
    cal_cat = "low"
    if c_score > 0.66:
        cal_cat = "high"
    elif c_score > 0.33:
        cal_cat = "medium"

    return {
        "caloric_density": float(c_score),
        "protein_index": float(p_score),
        "healthiness": float(h_score),
        "calorie_band": cal_cat,
    }


def dietary_tags(ings: List[str]) -> List[str]:
    s = " ".join(ings).lower()
    tags = []
    # Vegan/Vegetarian (very rough)
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

    # Spiciness signal
    if any(w in s for w in ["chili", "chilli", "jalapeno", "cayenne", "gochujang", "harissa", "pepper flakes"]):
        tags.append("spicy")

    return tags[:4]  # keep it compact


def generate_recipe(cuisine: str, ings: List[str]) -> str:
    """
    Structured, readable steps with numbered list.
    """
    title = f"{cuisine.title()} Style Dish with {', '.join(ings[:3]).title() if ings else 'Seasonal Ingredients'}"
    steps = [
        "Prep: finely chop aromatics (garlic/ginger/onion) and measure spices.",
        "Season main ingredients with salt and pepper.",
        "Heat oil in a pan; bloom spices and aromatics until fragrant.",
        "Add main ingredients, sear lightly, then cook through.",
        "Adjust with acid (lemon/lime/vinegar) and fresh herbs.",
        "Taste, correct seasoning, and serve warm."
    ]

    # Cuisine-specific flourish
    flair = {
        "indian": "Finish with garam masala and fresh cilantro.",
        "chinese": "Balance with a 1–1 soy sauce and rice vinegar splash; add sesame oil off-heat.",
        "italian": "Deglaze with a touch of white wine; finish with olive oil and basil.",
        "mexican": "Add cumin and chili powder; finish with lime and cilantro.",
        "japanese": "Season with a dash of mirin and soy; garnish with scallion.",
        "korean": "Stir in gochujang for heat and depth; top with sesame seeds.",
        "french": "Mount with a small knob of butter; add parsley and chives.",
        "thai": "Balance sweet–sour–salty with palm sugar, lime, and fish sauce."
    }
    steps.append(flair.get(cuisine.lower(), "Finish with fresh herbs and a drizzle of good olive oil."))

    block = f"{title}\n\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
    return block


def tts_bytes_en(text: str) -> bytes:
    """
    English TTS to MP3 bytes (gTTS). Single-voice for reliability on Streamlit Cloud.
    """
    tts = gTTS(text=text, lang="en")
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    return buf.getvalue()


def build_podcast_script(cuisine: str, ings: List[str], nutr: Dict[str, float], tags: List[str]) -> str:
    """
    Single-voice 'interview style' script (HOST / CHEF cues kept in text).
    """
    ttags = ", ".join(tags) if tags else "balanced"
    return textwrap.dedent(f"""
    HOST: Welcome to Quick Plates! Today we're exploring {cuisine.title()} cuisine.
    HOST: Our basket ingredients are: {', '.join(ings)}.

    CHEF: Great pick! This style often balances tradition with bold flavors.
    CHEF: For a quick home version, sauté aromatics, add your main ingredient, and finish with regional staples.

    HOST: Nutrition snapshot?
    CHEF: Caloric density looks {nutr['calorie_band']}, protein index {nutr['protein_index']:.2f}, overall healthiness {nutr['healthiness']:.2f}.
    CHEF: Dietary hints: {ttags}.

    HOST: Any final flourish?
    CHEF: Always taste and balance at the end. Fresh herbs and acid bring the dish alive.

    HOST: Thanks for cooking with us!""").strip()


# ------------------------ UI ------------------------
st.set_page_config(page_title=TITLE, page_icon="🍽️", layout="wide")
st.title(TITLE)
st.caption("TF–IDF + Logistic Regression • Multiclass cuisine classifier + recipe generator")

# Sidebar controls
st.sidebar.header("Settings")
enable_tts = st.sidebar.checkbox("Enable voice (English TTS)", value=False)
enable_podcast = st.sidebar.checkbox("Podcast mode (single-voice)", value=False)

with st.sidebar.expander("About the model", expanded=True):
    st.markdown("- Logistic Regression + TF-IDF (multiclass)")
pipe, INV = load_pipeline()
st.sidebar.markdown(f"- Classes: {len(INV)}")

# Input area
col_left, col_right = st.columns([1.2, 1.0], vertical_alignment="top")

with col_left:
    st.subheader("Ingredients")
    default_demo = "chicken, soy sauce, ginger, garlic, sesame oil"
    ing_text = st.text_area(
        "Comma-separated or one per line",
        value=default_demo,
        height=120,
        placeholder="e.g., tomato, basil, garlic, olive oil",
    )
    ings = pretty_ingredients_box(ing_text)

    if st.button("Find cuisine & build recipe", type="primary", use_container_width=True):
        if not ings:
            st.warning("Please provide at least one ingredient.")
        else:
            # Predict top-k + probabilities
            names, values, all_probs = predict_topk(pipe, INV, ings, k=TOPK)

            # Chart
            st.subheader("Top predictions")
            df_plot = pd.DataFrame({"cuisine": names, "probability": values})
            st.bar_chart(df_plot, x="cuisine", y="probability", use_container_width=True)

            # Top-1 block
            top1 = names[0]
            img_url = cuisine_image_url(top1)
            st.markdown(f"### Image • {top1.title()}")
            st.image(img_url, use_column_width=True)

            # Recipe text
            st.markdown("### 🧾 Generated Recipe")
            recipe_text = generate_recipe(top1, ings)
            st.text_area("Recipe", value=recipe_text, height=220, label_visibility="collapsed")

            # Nutrition + tags
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

            # Optional TTS: recipe reading
            if enable_tts:
                st.markdown("#### 🔊 Read recipe (English)")
                audio_bytes = tts_bytes_en(recipe_text)
                st.audio(audio_bytes, format="audio/mp3")

            # Optional Podcast mode (single voice for reliability)
            if enable_podcast:
                st.markdown("#### 🎙️ Auto-podcast")
                script = build_podcast_script(top1, ings, nutr, tags)
                st.text_area("Podcast transcript", value=script, height=180, label_visibility="collapsed")
                audio_pod = tts_bytes_en(script)
                st.audio(audio_pod, format="audio/mp3")

with col_right:
    st.subheader("How to use")
    st.markdown(
        """
        1. Enter ingredients (comma-separated or one per line).\n
        2. Click **Find cuisine & build recipe**.\n
        3. Review predictions, image, recipe, and nutrition hints.\n
        4. (Optional) Enable **voice** or **podcast** in the sidebar.
        """
    )

st.markdown("---")
st.caption("Tip: You can package this as a business demo (meal helper, shopping list recommender, or cooking coach).")

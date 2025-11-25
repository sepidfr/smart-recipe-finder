# app.py
# Smart Recipe Finder – Streamlit App
# - Classifies cuisine from ingredients using a pre-trained scikit-learn pipeline
# - Shows Top-3 predictions with probabilities
# - Fetches an illustrative food image (no API key needed)
# - Optional TTS (gTTS) to read the result in EN/FR/DE

from pathlib import Path
import json
import re
import io
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import requests
from gtts import gTTS

# ---------------------------------------------------------
# Paths (match your repo layout: files next to app.py)
# ---------------------------------------------------------
ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "cuisine_pipeline.joblib"
LABELS_PATH = ROOT / "labels.json"

# ---------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_pipeline():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    pipe = joblib.load(MODEL_PATH)

    if LABELS_PATH.exists():
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            labels = json.load(f)
    else:
        # Fallback: try to infer from classifier classes_
        try:
            labels = list(pipe.named_steps["clf"].classes_)
        except Exception:
            labels = None
    return pipe, labels

pipe, LABELS = load_pipeline()

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def normalize_token(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def ingredients_to_text(raw: str) -> str:
    """
    Accepts comma- or newline-separated ingredients from the textarea
    and converts to a single normalized string.
    """
    # split on comma or newline
    parts = re.split(r"[,\n]+", raw)
    parts = [normalize_token(p) for p in parts if p.strip()]
    return " ".join(parts)

def topk_prob(proba: np.ndarray, classes, k=3):
    idx = np.argsort(proba)[::-1][:k]
    return [(classes[i], float(proba[i])) for i in idx]

def cuisine_image_url(cuisine: str) -> str:
    """
    Keyless illustrative image via Unsplash source.
    """
    q = requests.utils.quote(f"{cuisine} cuisine food")
    return f"https://source.unsplash.com/800x600/?{q}"

def tts_bytes(text: str, lang_code: str = "en") -> bytes:
    """
    Generate MP3 bytes via gTTS (works on Streamlit Cloud).
    """
    tts = gTTS(text=text, lang=lang_code)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()

def example_dish(cuisine: str) -> str:
    """
    Minimal stub for a dish name per cuisine (no external dependency).
    """
    table = {
        "italian": "Spaghetti al Pomodoro",
        "mexican": "Tacos al Pastor",
        "indian": "Chicken Tikka Masala",
        "chinese": "Kung Pao Chicken",
        "japanese": "Chicken Teriyaki",
        "korean": "Bibimbap",
        "thai": "Pad Thai",
        "french": "Coq au Vin",
        "greek": "Greek Salad",
        "spanish": "Paella",
        "vietnamese": "Pho",
        "moroccan": "Chicken Tagine",
        "russian": "Borscht",
        "brazilian": "Feijoada",
        "british": "Fish and Chips",
        "cajun_creole": "Jambalaya",
        "filipino": "Adobo",
        "irish": "Beef Stew",
        "jamaican": "Jerk Chicken",
        "southern_us": "Fried Chicken",
    }
    return table.get(cuisine, f"{cuisine.title()} dish")

LANG2CODE = {"English": "en", "Français": "fr", "Deutsch": "de"}

# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.set_page_config(page_title="Smart Recipe Finder", page_icon="🍽️", layout="centered")
st.title("🍽️ Smart Recipe Finder")
st.caption("Classify cuisine from ingredients • Show top-3 • Image • Optional TTS")

with st.sidebar:
    st.header("Settings")
    lang_label = st.selectbox("Voice language (TTS)", list(LANG2CODE.keys()), index=0)
    do_tts = st.checkbox("Generate voice (TTS)", value=False)
    st.markdown("---")
    st.write("**About the model**")
    st.write("- TF-IDF + LogisticRegression (multiclass)")
    if LABELS is not None:
        st.write(f"- Classes: {len(LABELS)}")

st.subheader("Enter ingredients")
example = "chicken, soy sauce, ginger, garlic, sesame oil"
raw_ing = st.text_area("Comma or newline separated", value=example, height=120)

if st.button("Predict cuisine", type="primary"):
    if not raw_ing.strip():
        st.warning("Please enter at least one ingredient.")
        st.stop()

    text = ingredients_to_text(raw_ing)
    try:
        proba = pipe.predict_proba([text])[0]
        classes = getattr(pipe.named_steps["clf"], "classes_", LABELS)
        if classes is None:
            st.error("Could not determine class labels.")
            st.stop()
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # Top-3 table
    top3 = topk_prob(proba, classes, k=3)
    df_top = pd.DataFrame(
        [{"cuisine": c, "probability": p} for c, p in top3]
    ).assign(probability=lambda d: d["probability"].round(4))
    st.subheader("Predicted cuisines (Top-3)")
    st.dataframe(df_top, use_container_width=True, hide_index=True)

    # Bar chart
    st.bar_chart(df_top.set_index("cuisine"))

    # Representative image for top-1
    top1_cuisine, top1_p = top3[0]
    st.subheader(f"Image • {top1_cuisine.title()}")
    st.image(cuisine_image_url(top1_cuisine), caption=f"{top1_cuisine.title()} (illustrative)")

    # Example dish
    dish = example_dish(top1_cuisine)
    st.markdown(f"**Example dish:** {dish}")

    # Optional TTS
    if do_tts:
        lang_code = LANG2CODE[lang_label]
        speak_text = (
            f"Predicted cuisine: {top1_cuisine}. Example dish: {dish}. "
            f"Top three probabilities: " +
            ", ".join([f"{c}: {p:.2f}" for c, p in top3]) + "."
        )
        try:
            audio_bytes = tts_bytes(speak_text, lang_code=lang_code)
            st.audio(audio_bytes, format="audio/mp3")
        except Exception as e:
            st.warning(f"TTS failed: {e}")

st.markdown("---")
st.caption("© Smart Recipe Finder – TF-IDF + LogisticRegression pipeline")

import os
import io
import json
import joblib
import requests
import numpy as np
import streamlit as st
from pathlib import Path
from gtts import gTTS

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "cuisine_pipeline.joblib"
LABELS_PATH = APP_DIR / "labels.json"

@st.cache_resource(show_spinner=False)
def load_pipeline():
    pipe = joblib.load(MODEL_PATH)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)
    inv = {int(k): v for k, v in labels.items()}
    return pipe, inv

pipe, INV = load_pipeline()

# ---------------- Prediction ----------------
def predict_topk(pipe, inv, ingredients, k=3):
    text = " ".join(ingredients).lower()
    proba = pipe.predict_proba([text])[0]
    idx = np.argsort(proba)[::-1][:k]
    return [(inv[int(i)], float(proba[i])) for i in idx]

# ---------------- Recipe Generator ----------------
def generate_recipe(ingredients, cuisine):
    ing = [s.strip() for s in ingredients if s.strip()]
    title = f"{cuisine.title()} {ing[0] if ing else 'dish'}"
    steps = [
        "Prep ingredients by chopping aromatics such as garlic or onion.",
        "Season the main ingredients with salt, pepper, and spices.",
        "Heat oil in a pan and sauté aromatics until fragrant.",
        f"Add the main ingredients ({', '.join(ing)}) and cook thoroughly.",
        "Add herbs or sauces and let simmer for 3–5 minutes.",
        "Taste, adjust seasoning, and serve warm."
    ]
    return title, steps

# ----------- Image Fetch (no API key needed) -----------
def try_image(cuisine):
    try:
        q = requests.utils.quote(f"{cuisine} food dish")
        return f"https://source.unsplash.com/800x600/?{q}"
    except:
        return None

# -------------- TTS ----------------
def speak(text, lang="en"):
    buf = io.BytesIO()
    gTTS(text, lang=lang).write_to_fp(buf)
    buf.seek(0)
    return buf

# ---------------- Streamlit ----------------
st.set_page_config(page_title="Smart Recipe Finder", page_icon="🍳")

st.title("🍳 Smart Recipe Finder")
st.caption("TF-IDF + Logistic Regression | Predict cuisine + basic recipe + TTS")

with st.sidebar:
    st.header("Settings")
    do_tts = st.checkbox("Enable voice (TTS)", value=False)
    rate = st.slider("Voice speed (visual only, no effect)", 0.6, 1.4, 1.0, 0.05)
    st.write("Model: Logistic Regression")
    st.write("Classes: 20")

with st.expander("Enter ingredients", expanded=True):
    text = st.text_area(
        "Ingredients (comma-separated)",
        "chicken, soy sauce, ginger, garlic, sesame oil"
    )
    ingredients = [s.strip() for s in text.split(",")]

if st.button("Predict"):
    if len(ingredients) == 0:
        st.warning("Please enter ingredients.")
        st.stop()

    top3 = predict_topk(pipe, INV, ingredients, 3)
    top1, prob1 = top3[0]

    # --- Chart
    st.subheader("Predicted cuisines (Top 3)")
    st.bar_chart({"prob": [p for _, p in top3]}, x=None)

    # --- Image
    img = try_image(top1)
    if img:
        st.image(img, caption=f"{top1.title()}", use_column_width=True)

    # --- Recipe
    title, steps = generate_recipe(ingredients, top1)
    st.markdown(f"### 🍽️ {title}")
    recipe_text = title + "\n\n" + "\n".join([f"{i+1}. {s}" for i, s in enumerate(steps)])
    st.text_area("Recipe", recipe_text, height=220)

    # --- TTS
    if do_tts:
        audio = speak(recipe_text, "en")
        st.audio(audio, format="audio/mp3")

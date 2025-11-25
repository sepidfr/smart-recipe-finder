
import io
import os
import re
import json
import joblib
import numpy as np
import streamlit as st
from pathlib import Path

# Optional imports
try:
    from duckduckgo_search import DDGS
    HAVE_DDG = True
except Exception:
    HAVE_DDG = False

try:
    from gtts import gTTS
    from pydub import AudioSegment
    HAVE_TTS = True
except Exception:
    HAVE_TTS = False


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "cuisine_pipeline.joblib"
LABELS_PATH = APP_DIR / "models" / "labels.json"


@st.cache_resource
def load_pipeline():
    pipe = joblib.load(MODEL_PATH)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)
    return pipe, labels


pipe, LABELS = load_pipeline()
clf = pipe.named_steps["clf"]
tfidf = pipe.named_steps["tfidf"]


def normalize_ingredient(s):
    s = s.lower()
    s = re.sub(r"[^a-z\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def predict_topk(ingredients, k=3):
    text = " ".join(normalize_ingredient(x) for x in ingredients if x.strip())
    X = tfidf.transform([text])
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)[0]
        idx = np.argsort(proba)[-k:][::-1]
        return [(LABELS[i], float(proba[i])) for i in idx]
    else:
        y = clf.predict(X)[0]
        return [(LABELS[y], 1.0)]


def search_image(query):
    if not HAVE_DDG:
        return None
    try:
        with DDGS() as ddgs:
            results = ddgs.images(keywords=query, max_results=1)
            for r in results:
                return r.get("image") or r.get("thumbnail")
    except:
        return None
    return None


def speak_text(text, lang_code="en", rate=1.0):
    if not HAVE_TTS:
        return None
    tts = gTTS(text=text, lang=lang_code, slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    try:
        audio = AudioSegment.from_file(buf, format="mp3")
        new_rate = int(audio.frame_rate * float(rate))
        sped = audio._spawn(audio.raw_data, overrides={"frame_rate": new_rate}).set_frame_rate(audio.frame_rate)
        out = io.BytesIO()
        sped.export(out, format="mp3")
        out.seek(0)
        return out
    except:
        buf.seek(0)
        return buf


st.set_page_config(page_title="Recipe Finder", page_icon="🍳", layout="centered")

st.title("🍳 Recipe Finder — Cuisine Classifier")
st.caption("Enter ingredients, get cuisine predictions (Top-3), image, and optional TTS.")


ingredients_text = st.text_area(
    "Ingredients (one per line):",
    height=140,
    value="tomato\nbasil\nolıve oil\nparmesan",
)

k = st.slider("Top-K", 1, 5, 3, 1)
auto_img = st.checkbox("Auto Image", True)

lang = st.selectbox("Language", ["English", "Français", "Deutsch"], index=0)
lang_map = {"English": "en", "Français": "fr", "Deutsch": "de"}
lang_code = lang_map[lang]

if st.button("Predict", type="primary"):
    ingr = [x.strip() for x in ingredients_text.splitlines() if x.strip()]
    if not ingr:
        st.warning("Please enter ingredients.")
    else:
        preds = predict_topk(ingr, k=k)
        st.subheader("Top-K Predictions:")
        for i, (label, p) in enumerate(preds, 1):
            st.write(f"{i}. **{label}** — {p:.3f}")

        dish_name = preds[0][0].title() + " Dish"
        st.markdown(f"### {dish_name}")

        if auto_img:
            url = search_image(preds[0][0] + " cuisine dish")
            if url:
                st.image(url, caption=dish_name, use_container_width=True)

        st.session_state["speak_text"] = f"{dish_name}. Ingredients: " + ", ".join(ingr)


st.markdown("---")
st.subheader("Text-to-Speech")

default_tts = st.session_state.get("speak_text", "Predict a dish first.")
tts_input = st.text_area("Text:", value=default_tts, height=100)
rate = st.slider("Speed", 0.6, 1.6, 1.0, 0.05)

if st.button("Speak"):
    audio = speak_text(tts_input, lang_code, rate)
    if audio:
        st.audio(audio, format="audio/mp3")
    else:
        st.error("TTS not available.")

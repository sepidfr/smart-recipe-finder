# app.py
# ============================================================
# Smart Recipe Finder • Top-3 Cuisine Recipes + Duo-Voice Podcast
# Model: TF–IDF + Logistic Regression (joblib)
# UI: Streamlit
# Audio: ElevenLabs (male/female) if ELEVEN_API_KEY set; else gTTS fallback
# ============================================================

from __future__ import annotations
import io
import json
import os
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib
import streamlit as st

# Optional ElevenLabs; fallback to gTTS
ELEVEN_OK = False
try:
    from elevenlabs import VoiceSettings
    from elevenlabs.client import ElevenLabs
    ELEVEN_OK = True
except Exception:
    ELEVEN_OK = False

from gtts import gTTS  # fallback TTS

# -------------------- Paths & constants --------------------
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "cuisine_pipeline.joblib"
LABELS_PATH = APP_DIR / "labels.json"

TITLE = "Smart Recipe Finder"
TOPK = 3
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# -------------------- Cache: model & labels --------------------
@st.cache_resource(show_spinner="Loading model pipeline...")
def load_pipeline() -> Tuple[object, Dict[int, str]]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing {MODEL_PATH.name} in app root.")
    pipe = joblib.load(MODEL_PATH)

    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Missing {LABELS_PATH.name} in app root.")
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)
    if isinstance(labels, list):
        inv = {i: name for i, name in enumerate(labels)}
    elif isinstance(labels, dict):
        inv = {int(k): v for k, v in labels.items()}
    else:
        raise TypeError("labels.json must be a list or {index->name} dict.")
    return pipe, inv

# -------------------- Helpers --------------------
def cuisine_image_url(cuisine_name: str) -> str:
    q = f"{cuisine_name} plated dish"
    return f"https://source.unsplash.com/800x500/?{q.replace(' ', '%20')}"

def parse_ingredients(text: str) -> List[str]:
    parts = [t.strip() for t in text.replace("\n", ",").split(",")]
    return [p for p in parts if p]

def predict_topk(pipe, inv_labels: Dict[int, str], ingredients: List[str], k: int = TOPK):
    text = " ".join(ingredients)
    probs = pipe.predict_proba([text])[0]
    order = np.argsort(probs)[::-1][:k]
    names = [inv_labels[i] for i in order]
    values = probs[order]
    return names, values, probs

def qualitative_nutrition(ings: List[str]) -> Dict[str, float | str]:
    s = " ".join(ings).lower()
    hi_cal = sum(w in s for w in ["butter","oil","olive oil","ghee","cream","cheese","sugar","fried","bacon","nuts","peanut"])
    protein = sum(w in s for w in ["chicken","beef","pork","fish","tuna","egg","tofu","lentil","bean","chickpea","yogurt","paneer"])
    greens  = sum(w in s for w in ["spinach","kale","broccoli","herb","parsley","cilantro","tomato","cucumber","lettuce","carrot","zucchini","pepper"])

    c_score = min(1.0, 0.20*hi_cal + 0.15*len(ings))
    p_score = min(1.0, 0.12*protein + 0.02*len(ings))
    h_score = min(1.0, 0.10*greens + 0.02*(len(ings) - hi_cal))

    cal_band = "low"
    if c_score > 0.66: cal_band = "high"
    elif c_score > 0.33: cal_band = "medium"

    return {
        "caloric_density": float(c_score),
        "protein_index": float(p_score),
        "healthiness": float(h_score),
        "calorie_band": cal_band,
    }

def dietary_tags(ings: List[str]) -> List[str]:
    s = " ".join(ings).lower()
    tags = []
    animal  = any(w in s for w in ["chicken","beef","pork","fish","shrimp","egg","yogurt","cheese","milk","butter","honey"])
    dairy   = any(w in s for w in ["cheese","milk","yogurt","cream","butter","ghee"])
    gluten  = any(w in s for w in ["flour","bread","pasta","noodle","wheat","semolina","bulgur","couscous"])
    pork    = ("pork" in s) or ("bacon" in s) or ("ham" in s)
    alcohol = any(w in s for w in ["wine","beer","ale","vodka","rum","whiskey","brandy"])

    if not animal and not dairy: tags.append("vegan-ish")
    elif not animal:             tags.append("vegetarian-ish")
    if not gluten:               tags.append("gluten-light")
    if not pork:                 tags.append("pork-free")
    if not alcohol:              tags.append("no-alcohol")
    if any(w in s for w in ["chili","chilli","jalapeno","cayenne","gochujang","harissa","pepper flakes"]):
        tags.append("spicy")
    return tags[:4]

def food_value_score(nutr: Dict[str, float]) -> float:
    score = 0.35*nutr["protein_index"] + 0.40*nutr["healthiness"] - 0.25*nutr["caloric_density"]
    return float(np.clip(score, 0.0, 1.0))

def generate_recipe(cuisine: str, ings: List[str]) -> str:
    title = f"{cuisine.title()} Style with {', '.join(ings[:3]).title() if ings else 'Seasonal Ingredients'}"
    steps = [
        "Greet your guests and preheat the pan.",
        "Finely chop aromatics (garlic/ginger/onion) and measure spices.",
        "Season main ingredients with salt and pepper.",
        "Heat oil; bloom spices and aromatics until fragrant.",
        "Add main ingredients, sear, then cook through to tenderness.",
        "Balance with acid (lemon/lime/vinegar) and fresh herbs.",
        "Taste and correct seasoning; serve immediately."
    ]
    flourish = {
        "indian":  "Finish with garam masala and cilantro.",
        "chinese": "Add equal parts soy and rice vinegar; sesame oil off heat.",
        "italian": "Deglaze with white wine; finish with olive oil and basil.",
        "mexican": "Add cumin and chili; finish with lime and cilantro.",
        "japanese":"Season with mirin and soy; garnish with scallion.",
        "korean":  "Stir in gochujang; top with sesame seeds.",
        "french":  "Mount a small knob of butter; parsley/chives to finish.",
        "thai":    "Balance sweet–sour–salty with palm sugar, lime, fish sauce."
    }
    steps.append(flourish.get(cuisine.lower(), "Finish with fresh herbs and quality oil."))
    return f"{title}\n\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))

def build_podcast_duo(cuisine: str, ings: List[str], nutr: Dict[str, float], tags: List[str]) -> str:
    ttags = ", ".join(tags) if tags else "balanced"
    # Conversational, two personas (HOST=female, CHEF=male by default)
    return textwrap.dedent(f"""
    HOST: Hi everyone! Welcome to Quick Plates. I’m thrilled you’re here.
    HOST: Today we’re cooking in the spirit of {cuisine.title()}. Our basket holds {', '.join(ings)}.

    CHEF: Thanks for having me! This cuisine shines when you build flavor in layers—start with aromatics, then the main ingredient, and finish with bright accents.

    HOST: Let’s talk nutrition. What do you see?
    CHEF: Caloric density looks {nutr['calorie_band']}. Protein index {nutr['protein_index']:.2f}, healthiness {nutr['healthiness']:.2f}. Dietary hints: {ttags}.

    HOST: Any chef’s tip before we plate?
    CHEF: Taste at the end. A touch of acid and herbs will wake the dish up—keep it lively and balanced.

    HOST: Love it. Let’s cook!
    """).strip()

# -------------------- Audio backends --------------------
def tts_eleven(text: str, voice_id: str, model: str = "eleven_multilingual_v2") -> bytes:
    key = os.getenv("ELEVEN_API_KEY", "")
    if not (ELEVEN_OK and key):
        raise RuntimeError("ElevenLabs not configured.")
    client = ElevenLabs(api_key=key)
    audio = client.text_to_speech.convert(
        voice_id=voice_id,
        model_id=model,
        text=text,
        voice_settings=VoiceSettings(stability=0.5, similarity_boost=0.75, style=0.3, use_speaker_boost=True),
        output_format="mp3_44100_128"
    )
    buf = io.BytesIO()
    for chunk in audio:
        buf.write(chunk)
    return buf.getvalue()

def tts_gtts(text: str, accent: str = "US") -> bytes:
    # emulate persona by accent (tld) — male/female timbre requires a TTS provider
    tld_map = {"US":"com","UK":"co.uk","AU":"com.au"}
    tld = tld_map.get(accent.upper(), "com")
    tts = gTTS(text=text, lang="en", tld=tld)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    return buf.getvalue()

def voice_bytes(text: str, engine: str, voice_choice: str, persona: str) -> bytes:
    """
    engine: 'auto', 'eleven', or 'gtts'
    voice_choice:
        - if eleven: '<voice_id>'
        - if gtts:   'US'|'UK'|'AU'
    persona: 'HOST' or 'CHEF' (for potential styling hooks)
    """
    if engine == "auto":
        if ELEVEN_OK and os.getenv("ELEVEN_API_KEY", ""):
            engine = "eleven"
        else:
            engine = "gtts"
    if engine == "eleven":
        return tts_eleven(text, voice_id=voice_choice)
    else:
        return tts_gtts(text, accent=voice_choice)

# -------------------- UI --------------------
st.set_page_config(page_title=TITLE, page_icon="🍽️", layout="wide")
st.title(TITLE)
st.caption("Top-3 cuisine predictions • Three recipes • Duo-voice podcast • Nutrition & Food Value Score")

# Sidebar: TTS engine & voices
st.sidebar.header("Voice settings")
engine = st.sidebar.selectbox(
    "TTS engine",
    options=["auto", "eleven", "gtts"],
    index=0,
    help="Use ElevenLabs if API key is available; otherwise gTTS."
)

if engine in ("auto", "eleven") and ELEVEN_OK and os.getenv("ELEVEN_API_KEY", ""):
    st.sidebar.success("ElevenLabs available (male/female voices).")
    # You can pre-fill a few known voice IDs or let users paste their own
    st.sidebar.markdown("**Choose voices (male/female)**")
    host_voice = st.sidebar.text_input("HOST voice_id (female)", value="Rachel")   # name or id
    chef_voice = st.sidebar.text_input("CHEF voice_id (male)",   value="Adam")     # name or id
else:
    st.sidebar.info("Using gTTS fallback (accent styles).")
    host_voice = st.sidebar.selectbox("HOST accent", ["US", "UK", "AU"], index=2)
    chef_voice = st.sidebar.selectbox("CHEF accent", ["US", "UK", "AU"], index=0)

with st.sidebar.expander("About the model", expanded=True):
    st.markdown("- Logistic Regression over TF–IDF\n- Trained on Yummly ‘What’s Cooking?’\n- Deterministic and lightweight")

pipe, INV = load_pipeline()
st.sidebar.markdown(f"- Classes: **{len(INV)}**")

# Main input
left, right = st.columns([1.3, 1.0], vertical_alignment="top")
with left:
    st.subheader("Ingredients")
    demo = "chicken, soy sauce, ginger, garlic, sesame oil"
    ing_text = st.text_area("Comma-separated or one per line", value=demo, height=120)
    ings = parse_ingredients(ing_text)
    run = st.button("Predict & Build 3 Recipes", type="primary", use_container_width=True)

if run and not ings:
    st.warning("Please provide at least one ingredient.")

if run and ings:
    names, values, probs = predict_topk(pipe, INV, ings, k=TOPK)

    # Overview chart
    st.subheader("Top-3 Predictions")
    dfp = pd.DataFrame({"cuisine": names, "probability": values})
    st.bar_chart(dfp, x="cuisine", y="probability", use_container_width=True)

    # Three tabs: one per cuisine
    tabs = st.tabs([f"{i+1}) {c.title()}" for i, c in enumerate(names)])
    for idx, (tab, cuisine) in enumerate(zip(tabs, names)):
        with tab:
            colA, colB = st.columns([1.0, 1.0], vertical_alignment="top")
            with colA:
                st.image(cuisine_image_url(cuisine), caption=f"{cuisine.title()}", use_column_width=True)

            # Compose content
            recipe_text = generate_recipe(cuisine, ings)
            nutr = qualitative_nutrition(ings)
            tags = dietary_tags(ings)
            score = food_value_score(nutr)

            with colB:
                st.markdown("**Recipe**")
                st.text_area(" ", value=recipe_text, height=220, label_visibility="collapsed")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Caloric density", f"{nutr['caloric_density']:.2f}")
                c2.metric("Protein index", f"{nutr['protein_index']:.2f}")
                c3.metric("Healthiness", f"{nutr['healthiness']:.2f}")
                c4.metric("Food Value Score", f"{score:.2f}")
                st.caption(f"Dietary tags: {', '.join(tags) if tags else '—'}")

            # Duo-voice podcast (host ↔ chef)
            st.markdown("### 🎙️ Podcast (Host ↔ Chef)")
            script = build_podcast_duo(cuisine, ings, nutr, tags)
            st.text_area("Podcast Transcript", value=script, height=190)

            colR, colP = st.columns(2)
            with colR:
                if st.button(f"🔊 Read Recipe ({cuisine.title()})", key=f"rec_{idx}"):
                    try:
                        audio = voice_bytes(recipe_text, engine, host_voice, "HOST")
                        st.audio(audio, format="audio/mp3")
                    except Exception as e:
                        st.error(f"TTS error: {e}")

            with colP:
                if st.button(f"🎧 Play Podcast ({cuisine.title()})", key=f"pod_{idx}"):
                    try:
                        # Render host + chef separately and concatenate bytes if ElevenLabs; for simplicity we render as one block here
                        audio = voice_bytes(script, engine, chef_voice, "CHEF")
                        st.audio(audio, format="audio/mp3")
                    except Exception as e:
                        st.error(f"TTS error: {e}")

with right:
    st.subheader("How to use")
    st.markdown(
        "- Enter ingredients\n"
        "- Click **Predict & Build 3 Recipes**\n"
        "- Explore each cuisine tab: image, recipe, nutrition, value score\n"
        "- Use **🔊 Read Recipe** and **🎧 Play Podcast**\n"
        "- Configure voices in the sidebar"
    )

st.markdown("---")
st.caption("Deployable demo for meal planning / shopping assistant / cooking coach.")

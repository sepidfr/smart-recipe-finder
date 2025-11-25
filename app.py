# app.py — Smart Recipe Finder (Top-3 recipes • selectable voices • conversational podcast)
from __future__ import annotations
import io, json, re, textwrap, asyncio
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ============================== Optional audio deps ==============================
def _try_import_edge_tts():
    try:
        import edge_tts
        return edge_tts
    except Exception:
        return None

def _try_import_gtts():
    try:
        from gtts import gTTS
        return gTTS
    except Exception:
        return None

def _try_import_pydub():
    try:
        from pydub import AudioSegment
        return AudioSegment
    except Exception:
        return None

EDGE_TTS = _try_import_edge_tts()
gTTS     = _try_import_gtts()
AudioSeg = _try_import_pydub()

# ============================== Paths / constants ===============================
APP_DIR     = Path(__file__).resolve().parent
MODEL_PATH  = APP_DIR / "cuisine_pipeline.joblib"
LABELS_PATH = APP_DIR / "labels.json"

TITLE = "Smart Recipe Finder"
TOPK  = 3
np.random.seed(42)

# ============================== Nutrition per 100 g =============================
NUTR_TABLE = {
    "chicken":{"kcal":165,"protein":31.0,"fat":3.6,"carbs":0.0},
    "beef":{"kcal":217,"protein":26.1,"fat":11.8,"carbs":0.0},
    "pork":{"kcal":242,"protein":27.0,"fat":14.0,"carbs":0.0},
    "tofu":{"kcal":76,"protein":8.0,"fat":4.8,"carbs":1.9},
    "lentil":{"kcal":116,"protein":9.0,"fat":0.4,"carbs":20.1},
    "bean":{"kcal":127,"protein":8.7,"fat":0.5,"carbs":22.8},
    "chickpea":{"kcal":164,"protein":8.9,"fat":2.6,"carbs":27.4},
    "garlic":{"kcal":149,"protein":6.4,"fat":0.5,"carbs":33.1},
    "ginger":{"kcal":80,"protein":1.8,"fat":0.8,"carbs":17.8},
    "onion":{"kcal":40,"protein":1.1,"fat":0.1,"carbs":9.3},
    "tomato":{"kcal":18,"protein":0.9,"fat":0.2,"carbs":3.9},
    "basil":{"kcal":23,"protein":3.2,"fat":0.6,"carbs":2.7},
    "olive oil":{"kcal":884,"protein":0.0,"fat":100.0,"carbs":0.0},
    "sesame oil":{"kcal":884,"protein":0.0,"fat":100.0,"carbs":0.0},
    "butter":{"kcal":717,"protein":0.9,"fat":81.1,"carbs":0.1},
    "soy sauce":{"kcal":53,"protein":8.0,"fat":0.6,"carbs":5.6},
    "rice":{"kcal":130,"protein":2.4,"fat":0.3,"carbs":28.0},
    "pasta":{"kcal":131,"protein":5.0,"fat":1.1,"carbs":25.0},
    "yogurt":{"kcal":59,"protein":10.0,"fat":0.4,"carbs":3.6},
}

# ============================== Cache: model & labels ===========================
@st.cache_resource(show_spinner="Loading model pipeline...")
def load_pipeline():
    if not MODEL_PATH.exists():
        st.error(f"Missing model at {MODEL_PATH}"); st.stop()
    if not LABELS_PATH.exists():
        st.error(f"Missing labels at {LABELS_PATH}"); st.stop()
    pipe = joblib.load(MODEL_PATH)
    labels_raw = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    inv = {i: n for i, n in enumerate(labels_raw)} if isinstance(labels_raw, list) \
         else {int(k): v for k, v in labels_raw.items()}
    return pipe, inv

# ============================== Helpers ========================================
def cuisine_image_url(cuisine: str) -> str:
    return f"https://source.unsplash.com/800x500/?{(cuisine+' plated dish').replace(' ','%20')}"

def parse_ingredients(text: str) -> List[str]:
    raw = [t.strip() for t in text.replace("\n", ",").split(",")]
    return [t for t in raw if t]

def parse_mass_g(s: str, default: float = 100.0) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(g|gram|grams)\b", s.lower())
    return float(m.group(1)) if m else float(default)

def estimate_nutrition(items: List[str], default_mass_g: float = 100.0) -> Dict[str,float]:
    tot = {"kcal":0.0,"protein":0.0,"fat":0.0,"carbs":0.0}
    for it in items:
        it_l = it.lower()
        grams = parse_mass_g(it_l, default_mass_g)
        for key, nt in NUTR_TABLE.items():
            if key in it_l:
                mult = grams/100.0
                tot["kcal"]    += nt["kcal"]*mult
                tot["protein"] += nt["protein"]*mult
                tot["fat"]     += nt["fat"]*mult
                tot["carbs"]   += nt["carbs"]*mult
                break
    return tot

def predict_topk(pipe, inv_labels, ings: List[str], k: int = TOPK):
    text = " ".join(ings)
    proba = pipe.predict_proba([text])[0]
    order = np.argsort(proba)[::-1]
    # ensure distinct top-k classes
    order = order[:k]
    names = [inv_labels[i] for i in order]
    values = proba[order]
    return names, values

def qualitative_nutrition(ings: List[str]) -> Dict[str, float | str]:
    s = " ".join(ings).lower()
    hi_cal = sum(w in s for w in ["butter","oil","olive oil","ghee","cream","cheese","sugar","fried","bacon","nuts","peanut","sesame oil"])
    protein = sum(w in s for w in ["chicken","beef","pork","fish","tuna","egg","tofu","lentil","bean","chickpea","yogurt","paneer"])
    greens = sum(w in s for w in ["spinach","kale","broccoli","herb","parsley","cilantro","tomato","cucumber","lettuce","carrot","zucchini","pepper"])
    c_score = min(1.0, 0.20*hi_cal + 0.15*len(ings))
    p_score = min(1.0, 0.12*protein + 0.02*len(ings))
    h_score = min(1.0, 0.10*greens + 0.02*(len(ings)-hi_cal))
    cal_band = "low" if c_score <= 0.33 else ("medium" if c_score <= 0.66 else "high")
    return {"caloric_density":float(c_score),"protein_index":float(p_score),
            "healthiness":float(h_score),"calorie_band":cal_band}

def dietary_tags(ings: List[str]) -> List[str]:
    s = " ".join(ings).lower()
    tags = []
    animal = any(w in s for w in ["chicken","beef","pork","fish","shrimp","egg","yogurt","cheese","milk","butter","honey"])
    dairy  = any(w in s for w in ["cheese","milk","yogurt","cream","butter","ghee"])
    gluten = any(w in s for w in ["flour","bread","pasta","noodle","wheat","semolina","bulgur","couscous"])
    if not animal and not dairy: tags.append("vegan-ish")
    elif not animal: tags.append("vegetarian-ish")
    if not gluten: tags.append("gluten-light")
    if "pork" not in s and "bacon" not in s and "ham" not in s: tags.append("pork-free")
    if not any(w in s for w in ["wine","beer","ale","vodka","rum","whiskey","brandy"]): tags.append("no-alcohol")
    if any(w in s for w in ["chili","chilli","jalapeno","cayenne","gochujang","harissa","pepper flakes"]): tags.append("spicy")
    return tags[:4]

def food_value_score(n: Dict[str,float]) -> float:
    return float(np.clip(0.35*n["protein_index"] + 0.40*n["healthiness"] - 0.25*n["caloric_density"], 0.0, 1.0))

# ============================== Recipe generation ===============================
FLAIR = {
    "indian":"Finish with garam masala and fresh cilantro.",
    "chinese":"Add a 1–1 splash of soy sauce and rice vinegar; sesame oil off-heat.",
    "italian":"Deglaze with a touch of white wine; finish with olive oil and basil.",
    "mexican":"Add cumin and chili powder; finish with lime and cilantro.",
    "japanese":"Season with mirin and soy; garnish with scallion.",
    "korean":"Stir in gochujang; top with sesame seeds.",
    "french":"Mount with a small knob of butter; finish with parsley/chives.",
    "thai":"Balance sweet–sour–salty with palm sugar, lime, fish sauce."
}
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
    steps.append(FLAIR.get(cuisine.lower(), "Finish with fresh herbs and a drizzle of good olive oil."))
    return f"{title}\n\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))

# ============================== Conversational podcast ==========================
def build_podcast_dialogue(host_name: str, chef_name: str, cuisine: str, ings: List[str],
                           nutr: Dict[str,float], tags: List[str]) -> List[Tuple[str,str]]:
    """A natural Q&A (female host ↔ male chef) with greetings, humor, history, nutrition, and a brief method."""
    ttags = ", ".join(tags) if tags else "balanced"
    hcal = nutr["calorie_band"]
    pidx = f"{nutr['protein_index']:.2f}"
    hidx = f"{nutr['healthiness']:.2f}"

    lines = [
        ("HOST", f"Hello and welcome! I’m {host_name}, and today Chef {chef_name} is with us. Ready to cook some {cuisine.title()} magic?"),
        ("CHEF", "Hi there! Delighted to be here—let’s turn those pantry items into something memorable."),
        ("HOST", f"Here’s the basket: {', '.join(ings)}. What’s the game plan?"),
        ("CHEF", "We’ll bloom aromatics gently, build flavor with the staples of this cuisine, then finish bright and fresh."),
        ("HOST", f"Give us a quick historical nugget about {cuisine.title()} cuisine."),
        ("CHEF", f"{cuisine.title()} cooking balances tradition and regional produce; it evolved to highlight local flavors with simple techniques that are easy to love."),
        ("HOST", "Nutrition snapshot for our listeners?"),
        ("CHEF", f"Calorie density is {hcal}, protein index {pidx}, healthiness {hidx}. Dietary tags: {ttags}."),
        ("HOST", "Any chef’s tip we shouldn’t miss?"),
        ("CHEF", "Taste at the end and balance with acid and herbs—tiny adjustments make a dish sing. And keep heat moderate to protect aromatics."),
        ("HOST", "Perfect. Let’s cook!")
    ]
    return lines

# ============================== TTS helpers =====================================
# Edge-tts voices (female / male defaults + a few alternatives)
EDGE_FEMALE_DEFAULT = "en-US-JennyNeural"
EDGE_MALE_DEFAULT   = "en-US-GuyNeural"
EDGE_FEMALE_CHOICES = [
    "en-US-JennyNeural",
    "en-GB-LibbyNeural",
    "en-AU-NatashaNeural",
    "en-CA-ClaraNeural"
]
EDGE_MALE_CHOICES = [
    "en-US-GuyNeural",
    "en-GB-RyanNeural",
    "en-AU-WilliamNeural",
    "en-IN-PrabhatNeural"
]

async def _edge_tts_bytes_async(text: str, voice: str, rate: str = "+0%", volume: str = "+0%") -> bytes:
    tts = EDGE_TTS.Communicate(text=text, voice=voice, rate=rate, volume=volume)
    stream = io.BytesIO()
    async for chunk in tts.stream():
        if chunk["type"] == "audio":
            stream.write(chunk["data"])
    return stream.getvalue()

def tts_bytes_edge(text: str, voice: str) -> bytes | None:
    if EDGE_TTS is None:
        return None
    try:
        return asyncio.run(_edge_tts_bytes_async(text, voice))
    except Exception:
        return None

def tts_bytes_gtts(text: str, role: str) -> bytes | None:
    if gTTS is None:
        return None
    # Not truly gendered; use different TLDs to vary timbre slightly
    tld = "co.uk" if role == "HOST" else "com.au"
    try:
        tts = gTTS(text=text, lang="en", tld=tld)
        buf = io.BytesIO(); tts.write_to_fp(buf)
        return buf.getvalue()
    except Exception:
        return None

def tts_bytes(text: str, role: str, voice_name: str | None) -> bytes | None:
    """Try Edge TTS with selected voice; fallback to gTTS accent."""
    if voice_name and EDGE_TTS is not None:
        b = tts_bytes_edge(text, voice_name)
        if b: return b
    return tts_bytes_gtts(text, role)

def stitch_dialogue(dialogue: List[Tuple[str,str]], host_voice: str | None, chef_voice: str | None,
                    pause_ms: int = 300) -> bytes | None:
    """Return a single MP3 with alternating voices if pydub/ffmpeg available; else None."""
    if AudioSeg is None:
        return None
    try:
        track = AudioSeg.silent(duration=50)
        gap = AudioSeg.silent(duration=max(0, int(pause_ms)))
        for role, text in dialogue:
            voice = host_voice if role == "HOST" else chef_voice
            b = tts_bytes(text, role, voice)
            if not b:
                return None
            seg = AudioSeg.from_file(io.BytesIO(b), format="mp3")
            track += seg + gap
        out = io.BytesIO(); track.export(out, format="mp3")
        return out.getvalue()
    except Exception:
        return None

# ============================== UI =============================================
st.set_page_config(page_title=TITLE, page_icon="🍽️", layout="wide")
st.title(TITLE)
st.caption("Cuisine prediction • three recipe options • calories/macros • selectable voices • podcast")

# Sidebar: voice & podcast controls
st.sidebar.header("Audio Settings")
edge_available = EDGE_TTS is not None
st.sidebar.markdown(f"- Edge TTS available: **{'Yes' if edge_available else 'No (fallback to gTTS)'}**")

host_voice = None
chef_voice = None
if edge_available:
    host_voice = st.sidebar.selectbox("Host (female) voice", EDGE_FEMALE_CHOICES, index=0)
    chef_voice = st.sidebar.selectbox("Chef (male) voice",   EDGE_MALE_CHOICES,   index=0)
else:
    st.sidebar.info("Edge TTS not found: voices will fallback to gTTS accents (not truly gendered).")

enable_recipe_tts = st.sidebar.checkbox("Enable voice for selected recipe", value=True)
enable_podcast    = st.sidebar.checkbox("Enable conversational podcast (Host ↔ Chef)", value=True)
podcast_pause     = st.sidebar.slider("Pause between dialogue turns (ms)", 150, 800, 300, step=50)
host_name         = st.sidebar.text_input("Host display name", value="Sara")
chef_name         = st.sidebar.text_input("Chef display name", value="Masoud")

with st.sidebar.expander("About the model", expanded=False):
    st.markdown("- Logistic Regression over TF–IDF features\n- Trained on the Yummly ‘What’s Cooking?’ dataset")

# Load model/labels
pipe, INV = load_pipeline()
st.sidebar.markdown(f"- Classes: **{len(INV)}**")

# Main layout
left, right = st.columns([1.25, 1.0], vertical_alignment="top")

with left:
    st.subheader("Ingredients")
    demo = "chicken, soy sauce, ginger, garlic, sesame oil"
    ing_text = st.text_area("Comma-separated or one per line", value=demo, height=120,
                            placeholder="e.g., tomato, basil, garlic, olive oil")
    ings = parse_ingredients(ing_text)

    run = st.button("Predict cuisines & build 3 recipe options", type="primary", use_container_width=True)
    if run:
        if not ings:
            st.warning("Please provide at least one ingredient.")
            st.stop()

        # Top-3 distinct cuisines
        cuisines, probs = predict_topk(pipe, INV, ings, k=TOPK)
        df_pred = pd.DataFrame({"cuisine": cuisines, "probability": probs})
        st.markdown("### Top predictions")
        st.bar_chart(df_pred.set_index("cuisine")["probability"], use_container_width=True)

        # Build options (one per cuisine)
        options_meta = {}
        for c in cuisines:
            recipe = generate_recipe(c, ings)
            macro  = estimate_nutrition(ings, default_mass_g=100.0)
            nutr   = qualitative_nutrition(ings)
            fvs    = food_value_score(nutr)
            options_meta[c] = {"recipe": recipe, "macro": macro, "nutr": nutr, "fvs": fvs}

        # Tabs to preview all three recipes
        st.markdown("### Explore three recipe options")
        tabs = st.tabs([f"{c.title()} • {int(options_meta[c]['macro']['kcal']):d} kcal • FVS {options_meta[c]['fvs']:.2f}" for c in cuisines])
        for tab, c in zip(tabs, cuisines):
            with tab:
                st.markdown(f"**Cuisine:** {c.title()}")
                st.image(cuisine_image_url(c), use_column_width=True)
                st.text_area("Recipe preview", value=options_meta[c]["recipe"], height=200, label_visibility="collapsed")
                m, n = options_meta[c]["macro"], options_meta[c]["nutr"]
                a,b,c3,c4 = st.columns(4)
                a.metric("Calories", f"{m['kcal']:.0f} kcal")
                b.metric("Protein",  f"{m['protein']:.1f} g")
                c3.metric("Fat",     f"{m['fat']:.1f} g")
                c4.metric("Carbs",   f"{m['carbs']:.1f} g")
                st.caption(f"Caloric density (0–1): {n['caloric_density']:.2f} • "
                           f"Protein index (0–1): {n['protein_index']:.2f} • "
                           f"Healthiness (0–1): {n['healthiness']:.2f}")

        # Selection for voice/podcast target
        st.markdown("### Choose a recipe for voice & podcast")
        display_labels = [f"{c.title()} • {int(options_meta[c]['macro']['kcal']):d} kcal • FVS {options_meta[c]['fvs']:.2f}" for c in cuisines]
        label_to_cuisine = {lab: c for lab, c in zip(display_labels, cuisines)}
        selected_label = st.selectbox("Select one:", display_labels, index=0, label_visibility="collapsed")
        sel_cuisine = label_to_cuisine[selected_label]
        sel = options_meta[sel_cuisine]

        # Voice: single-voice read of the recipe (host voice by default)
        if enable_recipe_tts:
            st.markdown("#### 🔊 Voice recipe")
            recipe_audio = tts_bytes(sel["recipe"], role="HOST", voice_name=host_voice)
            if recipe_audio:
                st.audio(recipe_audio, format="audio/mp3")
            else:
                st.info("TTS unavailable in this environment.")

        # Conversational podcast: female host ↔ male chef (greetings + Q&A)
        if enable_podcast:
            st.markdown("#### 🎙️ Conversational podcast (Host ↔ Chef)")
            dialogue = build_podcast_dialogue(host_name, chef_name, sel_cuisine, ings, sel["nutr"], dietary_tags(ings))
            st.markdown("\n".join([f"**{r}:** {t}" for r,t in dialogue]))

            combined = stitch_dialogue(dialogue, host_voice, chef_voice, pause_ms=int(podcast_pause))
            if combined:
                st.audio(combined, format="audio/mp3")
            else:
                st.caption("Fallback: playing each turn separately.")
                for i, (role, text) in enumerate(dialogue, 1):
                    vname = host_voice if role == "HOST" else chef_voice
                    b = tts_bytes(text, role, vname)
                    if b:
                        st.markdown(f"*Turn {i} — {role}*")
                        st.audio(b, format="audio/mp3")
                    else:
                        st.info("TTS unavailable in this environment.")
                        break

with right:
    st.subheader("How to use")
    st.markdown(
        "1) Enter ingredients\n\n"
        "2) Click **Predict cuisines & build 3 recipe options**\n\n"
        "3) Preview three tabs (one per predicted cuisine)\n\n"
        "4) Choose a recipe for **voice** and **podcast**\n\n"
        "5) Adjust **voices** (female host / male chef) in the sidebar"
    )

st.markdown("---")
st.caption("Meal helper • shopping assistant • cooking coach")

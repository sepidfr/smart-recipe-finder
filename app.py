# app.py — Smart Recipe Finder (full, professional, English)
# ----------------------------------------------------------
# - TF–IDF + Logistic Regression (joblib pipeline)
# - Predict TOP-3 cuisines → show 3 recipe options (distinct cuisines, with fallback)
# - One selection drives: preview, charts, single-voice TTS, and a HOST↔CHEF podcast
# - Robust TTS: Edge TTS (male/female real voices) → gTTS fallback (accents via TLD)
# - Safe asyncio handling for Edge TTS; optional MP3 stitch via PyDub + ffmpeg
# - Unsplash image fallback; download buttons for podcast MP3 and recipe text

from __future__ import annotations
import io
import re
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go

# ─────────────────────────── Optional audio deps ───────────────────────────────
def _has_edge_tts() -> bool:
    try:
        import edge_tts  # noqa: F401
        return True
    except Exception:
        return False

def _has_pydub_ffmpeg() -> bool:
    try:
        from pydub import AudioSegment  # noqa: F401
        return True
    except Exception:
        return False

EDGE_OK  = _has_edge_tts()
PYDUB_OK = _has_pydub_ffmpeg()

# ─────────────────────────── Paths / constants ─────────────────────────────────
APP_DIR     = Path(__file__).resolve().parent
MODEL_PATH  = APP_DIR / "cuisine_pipeline.joblib"
LABELS_PATH = APP_DIR / "labels.json"

TITLE = "Smart Recipe Finder"
TOPK  = 3
np.random.seed(42)

# ─────────────────────── Nutrition per 100 g (simple table) ───────────────────
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

# ───────────────────────────── Cache: model & labels ───────────────────────────
@st.cache_resource(show_spinner="Loading model pipeline...")
def load_pipeline():
    if not MODEL_PATH.exists():
        st.error(f"Missing model at {MODEL_PATH}. Upload your TF–IDF+LogReg joblib file.")
        st.stop()
    if not LABELS_PATH.exists():
        st.error(f"Missing labels at {LABELS_PATH}. Provide a JSON list or {idx: name} mapping.")
        st.stop()
    pipe = joblib.load(MODEL_PATH)
    labels_raw = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    inv = {i: n for i, n in enumerate(labels_raw)} if isinstance(labels_raw, list) \
         else {int(k): v for k, v in labels_raw.items()}
    return pipe, inv

# ──────────────────────────────── Helpers ─────────────────────────────────────
def cuisine_image_url(cuisine: str) -> str:
    # Unsplash random image by query; fallback is handled by Streamlit automatically if unreachable
    q = (cuisine + " plated dish").replace(" ", "%20")
    return f"https://source.unsplash.com/800x500/?{q}"

def parse_ingredients(text: str) -> List[str]:
    raw = [t.strip() for t in text.replace("؛", ",").replace("\n", ",").split(",")]
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

def qualitative_nutrition(ings: List[str]) -> Dict[str, float | str]:
    s = " ".join(ings).lower()
    hi_cal = sum(w in s for w in ["butter","oil","olive oil","ghee","cream","cheese","sugar","fried","bacon","nuts","peanut","sesame oil"])
    protein = sum(w in s for w in ["chicken","beef","pork","fish","tuna","egg","tofu","lentil","bean","chickpea","yogurt","paneer"])
    greens = sum(w in s for w in ["spinach","kale","broccoli","herb","parsley","cilantro","tomato","cucumber","lettuce","carrot","zucchini","pepper"])
    c_score = float(np.clip(0.20*hi_cal + 0.15*len(ings), 0.0, 1.0))
    p_score = float(np.clip(0.12*protein + 0.02*len(ings), 0.0, 1.0))
    h_score = float(np.clip(0.10*greens + 0.02*(len(ings)-hi_cal), 0.0, 1.0))
    cal_band = "low" if c_score <= 0.33 else ("medium" if c_score <= 0.66 else "high")
    return {"caloric_density":c_score,"protein_index":p_score,"healthiness":h_score,"calorie_band":cal_band}

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
    if any(w in s for w in ["chili","chilli","jalapeno","cayenne","gochujang","harissa","pepper flakes","hot","spicy"]): tags.append("spicy")
    return tags[:4]

def food_value_score(n: Dict[str,float]) -> float:
    return float(np.clip(0.35*n["protein_index"] + 0.40*n["healthiness"] - 0.25*n["caloric_density"], 0.0, 1.0))

# ───────────────────────── Recipe generation (deterministic) ──────────────────
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

def _best_3(ings: List[str]) -> str:
    # Compact ingredient phrase for titles
    base = ", ".join([i.strip() for i in ings if i.strip()][:3])
    return base if base else "Seasonal Ingredients"

def generate_recipe(cuisine: str, ings: List[str]) -> str:
    title = f"{cuisine.title()}-Style Dish with {_best_3(ings).title()}"
    # A fuller, step-by-step script that later can be read inside the podcast
    steps = [
        "Set up: wash and dry produce. Finely chop garlic, ginger, and onion; measure spices.",
        "Season mains lightly with salt and black pepper; keep at room temperature 10 minutes.",
        "Pan on medium heat. Add oil; bloom aromatics 60–90 seconds until fragrant (no browning).",
        "Add mains; sear 2–3 minutes to develop light color, then reduce heat to medium-low.",
        "Layer flavor: add core spices and a splash of stock or water. Simmer until tender.",
        "Adjust: a touch of acid (lemon/lime/vinegar) and fresh herbs for brightness.",
        "Rest 2 minutes; plate and finish with a drizzle of good oil or fresh herbs."
    ]
    steps.append(FLAIR.get(cuisine.lower(), "Finish with fresh herbs and a drizzle of olive oil."))
    body = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
    return f"{title}\n\n{body}"

# ─────────────────────────── Prediction (robust TOP-3) ────────────────────────
def predict_topk(pipe, inv_labels: Dict[int,str], ings: List[str], k: int = TOPK) -> Tuple[List[str], np.ndarray]:
    text = " ".join(ings)
    proba = pipe.predict_proba([text])[0]
    order = np.argsort(proba)[::-1]
    names, values, seen = [], [], set()
    for idx in order:
        c = inv_labels[idx]
        if c not in seen:
            names.append(c); values.append(proba[idx]); seen.add(c)
        if len(names) == k:
            break
    # Fallback, ensure distinct k classes even if model is uncertain
    if len(names) < k:
        FALLBACK = ["italian","mexican","indian","chinese","french","japanese","thai","korean","spanish","greek"]
        for c in FALLBACK:
            if len(names) == k: break
            if c not in seen:
                names.append(c); values.append(0.0); seen.add(c)
    return names, np.array(values, dtype=float)

# ─────────────────────────────── TTS Utilities ────────────────────────────────
EDGE_FEMALE_CHOICES = ["en-US-JennyNeural","en-GB-LibbyNeural","en-AU-NatashaNeural","en-CA-ClaraNeural"]
EDGE_MALE_CHOICES   = ["en-US-GuyNeural","en-GB-RyanNeural","en-AU-WilliamNeural","en-IN-PrabhatNeural"]

GTTs_ACCENTS = {
    "US": "com",
    "UK": "co.uk",
    "AU": "com.au",
    "CA": "ca",
    "IN": "co.in",
}

def _ssml_wrap_chat(text: str, rate: str = "+0%", pitch: str = "+0%") -> str:
    safe = (text or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    return f"""<speak version="1.0" xml:lang="en-US">
  <mstts:express-as style="chat" styledegree="2" xmlns:mstts="https://www.w3.org/2001/mstts">
    <prosody rate="{rate}" pitch="{pitch}">{safe}</prosody>
  </mstts:express-as>
</speak>"""

async def _edge_ssml_to_bytes_async(ssml: str, voice: str) -> bytes:
    import edge_tts
    tts = edge_tts.Communicate(ssml=ssml, voice=voice)
    buf = io.BytesIO()
    async for chunk in tts.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    return buf.getvalue()

def _run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Run on a dedicated loop to avoid "event loop already running"
            new_loop = asyncio.new_event_loop()
            try:
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        # No loop at all
        return asyncio.run(coro)

def tts_bytes_any(
    text: str,
    role: str,
    edge_voice_name: Optional[str],
    rate: str = "+0%",
    pitch: str = "+0%",
    gtts_tld: str = "com",
) -> Optional[bytes]:
    # Try Edge first if a voice name is provided and Edge is available
    if edge_voice_name and EDGE_OK:
        try:
            ssml = _ssml_wrap_chat(text, rate=rate, pitch=pitch)
            return _run_async(_edge_ssml_to_bytes_async(ssml, edge_voice_name))
        except Exception:
            pass  # fall back to gTTS below

    # gTTS fallback (not truly gendered, but accent differs by TLD)
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang="en", tld=gtts_tld)
        out = io.BytesIO()
        tts.write_to_fp(out)
        return out.getvalue()
    except Exception:
        return None

def stitch_dialogue(
    dialogue: List[Tuple[str,str]],
    host_voice: Optional[str],
    chef_voice: Optional[str],
    pause_ms: int = 250,
    rate: str = "+0%",
    pitch: str = "+0%",
    tld_host: str = "com",
    tld_chef: str = "com"
) -> Optional[bytes]:
    if not PYDUB_OK:
        return None
    try:
        from pydub import AudioSegment
        track = AudioSegment.silent(duration=60)
        gap   = AudioSegment.silent(duration=max(0, int(pause_ms)))
        for role, text in dialogue:
            voice = host_voice if role == "HOST" else chef_voice
            tld   = tld_host if role == "HOST" else tld_chef
            b = tts_bytes_any(text, role, voice, rate=rate, pitch=pitch, gtts_tld=tld)
            if not b:
                return None
            seg = AudioSegment.from_file(io.BytesIO(b), format="mp3")
            track += seg + gap
        out = io.BytesIO()
        track.export(out, format="mp3")
        return out.getvalue()
    except Exception:
        return None

# ───────────────────────────── Podcast (dialogue) ─────────────────────────────
def _clean_recipe_lines(recipe_text: str) -> List[str]:
    # Extract lines that look like steps; keep both "1. ..." and plain lines
    lines = [ln.strip() for ln in recipe_text.split("\n") if ln.strip()]
    # Remove title line
    if lines and "-Style Dish" in lines[0]:
        lines = lines[1:]
    # Keep lines that are either numbered or sentences
    cleaned = []
    for ln in lines:
        # strip leading numbering "1. " etc.
        cleaned.append(re.sub(r"^\d+\.\s*", "", ln))
    return cleaned

def build_podcast_dialogue(
    host_name: str,
    chef_name: str,
    cuisine: str,
    ings: List[str],
    nutr: Dict[str,float|str],
    tags: List[str],
    recipe_text: str
) -> List[Tuple[str,str]]:
    base = ", ".join(ings[:2]) if ings else "basic pantry items"
    ttags = ", ".join(tags) if tags else "balanced"
    hcal  = nutr["calorie_band"]; pidx = f"{float(nutr['protein_index']):.2f}"; hidx = f"{float(nutr['healthiness']):.2f}"

    # Conversational intro
    dlg: List[Tuple[str,str]] = [
        ("HOST", f"Hello everyone — {host_name} here, and welcome back to Flavor Talks!"),
        ("HOST", f"Today I'm joined by Chef {chef_name}, bringing modern {cuisine.title()} flavors."),
        ("CHEF", f"Hi {host_name}, thanks for having me. I love this basket — {base} gives us options."),
        ("HOST", f"In one breath, what defines {cuisine.title()} cuisine to you?"),
        ("CHEF", f"It’s heritage meeting balance. Even with {base}, you can feel its regional backbone."),
        ("HOST", "Give us a quick nutrition postcard for the dish we’ll build."),
        ("CHEF", f"Calorie density {hcal}; protein index {pidx}; healthiness {hidx}. Tags: {ttags}."),
        ("HOST", "Great — now walk us through the actual step-by-step. Keep it crisp."),
    ]

    # Insert recipe steps as chef speech
    steps = _clean_recipe_lines(recipe_text)
    step_no = 1
    for ln in steps:
        if not ln: 
            continue
        dlg.append(("CHEF", f"Step {step_no}: {ln}"))
        step_no += 1

    # Close
    dlg.extend([
        ("HOST", "Beautiful — fast, clear, and packed with flavor."),
        ("CHEF", "Thanks! Tiny acid and fresh herbs at the end — that’s the glow."),
    ])
    return dlg

# ───────────────────────────────── UI ─────────────────────────────────────────
st.set_page_config(page_title=TITLE, page_icon="🍽️", layout="wide")
st.title(TITLE)
st.caption("Cuisine prediction • three recipe options • calories/macros • selectable voices • conversational podcast")

# Sidebar: audio config
st.sidebar.header("Audio Settings")
st.sidebar.markdown(f"- Edge TTS available: **{'Yes' if EDGE_OK else 'No (fallback to gTTS)'}**")
st.sidebar.markdown(f"- MP3 stitch (PyDub/ffmpeg): **{'Yes' if PYDUB_OK else 'No'}**")

# Voice pickers
if EDGE_OK:
    host_voice = st.sidebar.selectbox("Host voice (female)", EDGE_FEMALE_CHOICES, index=0, key="host_voice_sel")
    chef_voice = st.sidebar.selectbox("Chef voice (male)",   EDGE_MALE_CHOICES,   index=0, key="chef_voice_sel")
    tld_host = "com"; tld_chef = "com"  # Irrelevant when Edge is on, kept for API completeness
else:
    st.sidebar.info("Edge voices not available — using gTTS accents (not truly gendered).")
    host_voice = None
    chef_voice = None
    # Let user choose accents to simulate voice difference
    host_acc = st.sidebar.selectbox("Host accent (gTTS)", list(GTTs_ACCENTS.keys()), index=1, key="host_acc")
    chef_acc = st.sidebar.selectbox("Chef accent (gTTS)", list(GTTs_ACCENTS.keys()), index=0, key="chef_acc")
    tld_host = GTTs_ACCENTS[host_acc]
    tld_chef = GTTs_ACCENTS[chef_acc]

voice_rate        = st.sidebar.selectbox("Voice speed", ["-10%","-5%","+0%","+5%","+10%"], index=2, key="rate_sel")
voice_pitch       = st.sidebar.selectbox("Voice pitch", ["-2%","+0%","+2%","+4%"], index=1, key="pitch_sel")
enable_recipe_tts = st.sidebar.checkbox("Enable voice for selected recipe", value=True, key="rec_tts_chk")
enable_podcast    = st.sidebar.checkbox("Enable conversational podcast (Host ↔ Chef)", value=True, key="podcast_chk")
podcast_pause     = st.sidebar.slider("Pause between turns (ms)", 150, 800, 300, step=50, key="pause_sel")
host_name         = st.sidebar.text_input("Host display name", value="Sara", key="host_name_in")
chef_name         = st.sidebar.text_input("Chef display name", value="Masoud", key="chef_name_in")

with st.sidebar.expander("About the model", expanded=False):
    st.markdown("- Logistic Regression over TF–IDF features\n- Trained on the Yummly ‘What’s Cooking?’ dataset")

# Load model/labels
pipe, INV = load_pipeline()
st.sidebar.markdown(f"- Classes: **{len(INV)}**")

# ─────────────────────────────── Layout ───────────────────────────────────────
left, right = st.columns([1.25, 1.0])

# Input
with left:
    st.subheader("Ingredients")
    demo = "chicken, soy sauce, ginger, garlic, sesame oil"
    ing_text = st.text_area(
        "Comma-separated or one per line",
        value=demo,
        height=120,
        placeholder="e.g., tomato, basil, garlic, olive oil",
        key="ing_text"
    )
    ings = parse_ingredients(ing_text)

    run = st.button("Predict cuisines & build 3 recipe options", type="primary", use_container_width=True, key="predict_btn")
    if run:
        if not ings:
            st.warning("Please provide at least one ingredient.")
            st.stop()

        cuisines, probs = predict_topk(pipe, INV, ings, k=TOPK)
        df_pred = pd.DataFrame({"cuisine": cuisines, "probability": probs})

        options_meta: Dict[str, Dict] = {}
        for c in cuisines:
            recipe = generate_recipe(c, ings)
            macro  = estimate_nutrition(ings, default_mass_g=100.0)
            nutr   = qualitative_nutrition(ings)
            fvs    = food_value_score(nutr)
            options_meta[c] = {"recipe": recipe, "macro": macro, "nutr": nutr, "fvs": fvs}

        st.session_state["pred_ready"]       = True
        st.session_state["df_pred"]          = df_pred
        st.session_state["cuisines"]         = cuisines
        st.session_state["options_meta"]     = options_meta
        st.session_state["selected_cuisine"] = cuisines[0]    # default = top-1
        st.rerun()

# Render
with left:
    if st.session_state.get("pred_ready", False):
        df_pred      = st.session_state["df_pred"]
        cuisines     = st.session_state["cuisines"]
        options_meta = st.session_state["options_meta"]
        selected     = st.session_state.get("selected_cuisine", cuisines[0])

        # Top predictions chart
        st.markdown("### Top predictions")
        fig_pred = go.Figure(data=[go.Bar(x=df_pred["cuisine"], y=df_pred["probability"])])
        fig_pred.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                               yaxis=dict(title="Probability", rangemode="tozero"),
                               xaxis=dict(title="Cuisine"), height=300, template="simple_white")
        st.plotly_chart(fig_pred, use_container_width=True, config={"displayModeBar": False}, key="pred_chart")

        # Single source of truth: radio drives the rest
        st.markdown("### Explore three recipe options")
        chosen = st.radio(
            "Pick one to preview & voice:",
            options=cuisines,
            index=cuisines.index(selected),
            horizontal=True,
            label_visibility="collapsed",
            key="recipe_radio",
        )
        if chosen != selected:
            st.session_state["selected_cuisine"] = chosen
            selected = chosen

        # Preview for the selected recipe only
        c = selected
        st.markdown(f"**Cuisine:** {c.title()}")
        st.image(cuisine_image_url(c), use_column_width=True)
        recipe_text = options_meta[c]["recipe"]
        st.text_area("Recipe", value=recipe_text, height=260, label_visibility="collapsed", key=f"recipe_preview_{c}")

        # Macro chart
        m = options_meta[c]["macro"]
        macro_names = ["Calories (kcal)", "Protein (g)", "Fat (g)", "Carbs (g)"]
        macro_vals  = [m["kcal"], m["protein"], m["fat"], m["carbs"]]
        fig_macro = go.Figure(data=[go.Bar(x=macro_names, y=macro_vals)])
        fig_macro.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                                yaxis=dict(title="Amount", rangemode="tozero"),
                                xaxis=dict(title=""), height=300, template="simple_white")
        st.plotly_chart(fig_macro, use_container_width=True, config={"displayModeBar": False}, key=f"macro_{c}")

        # Value chart
        n = options_meta[c]["nutr"]
        value_names = ["Caloric density", "Protein index", "Healthiness", "Food Value Score"]
        value_vals  = [n["caloric_density"], n["protein_index"], n["healthiness"], options_meta[c]["fvs"]]
        fig_val = go.Figure(data=[go.Bar(x=value_names, y=value_vals)])
        fig_val.update_yaxes(range=[0, 1])
        fig_val.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                              yaxis=dict(title="Score (0–1)", rangemode="tozero"),
                              xaxis=dict(title=""), height=300, template="simple_white")
        st.plotly_chart(fig_val, use_container_width=True, config={"displayModeBar": False}, key=f"value_{c}")

        # Voice: single-file recipe TTS (host voice)
        if enable_recipe_tts:
            st.markdown("#### 🔊 Voice recipe")
            tld_for_host = tld_host if not EDGE_OK else "com"
            audio = tts_bytes_any(recipe_text, role="HOST", edge_voice_name=host_voice,
                                  rate=voice_rate, pitch=voice_pitch, gtts_tld=tld_for_host)
            if audio:
                st.audio(audio, format="audio/mp3")
                st.download_button("Download recipe MP3", data=audio,
                                   file_name=f"{c}_recipe.mp3", mime="audio/mpeg", key=f"dl_recipe_{c}")
            else:
                st.info("TTS unavailable in this environment (Edge blocked and/or gTTS missing).")

        # Podcast: conversational, includes REAL step-by-step recipe
        if enable_podcast:
            st.markdown("#### 🎙️ Conversational podcast (Host ↔ Chef)")
            dlg = build_podcast_dialogue(host_name, chef_name, c, ings, n, dietary_tags(ings), recipe_text)
            st.markdown("\n".join([f"**{r}:** {t}" for r, t in dlg]))

            stitched = stitch_dialogue(
                dlg, host_voice, chef_voice,
                pause_ms=int(podcast_pause), rate=voice_rate, pitch=voice_pitch,
                tld_host=tld_host, tld_chef=tld_chef
            )
            if stitched:
                st.audio(stitched, format="audio/mp3")
                st.download_button("Download podcast MP3", data=stitched,
                                   file_name=f"{c}_podcast.mp3", mime="audio/mpeg", key=f"dl_podcast_{c}")
            else:
                st.caption("Single-file stitch unavailable — playing turn-by-turn.")
                # Turn-by-turn playback
                for i, (role, text) in enumerate(dlg, 1):
                    vname = host_voice if role == "HOST" else chef_voice
                    tld   = tld_host if role == "HOST" else tld_chef
                    b = tts_bytes_any(text, role, vname, rate=voice_rate, pitch=voice_pitch, gtts_tld=tld)
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
        "3) Use the **radio** to pick one of the three cuisines (selection syncs everywhere)\n\n"
        "4) Turn on **Voice** and/or **Podcast** in the sidebar and set voices\n\n"
        "5) Download the MP3 files if you like"
    )

    st.markdown("---")
    st.subheader("Notes")
    st.markdown(
        "- Ensure you provide **cuisine_pipeline.joblib** and **labels.json** in the app folder.\n"
        "- For real female/male voices, Edge TTS must be available. Otherwise, gTTS accents are used.\n"
        "- MP3 stitching requires **pydub** and **ffmpeg** to be installed in the environment."
    )

st.markdown("---")
st.caption("Meal helper • shopping assistant • cooking coach")

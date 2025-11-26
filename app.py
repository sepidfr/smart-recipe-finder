# app.py — Smart Recipe Finder (professional, fixed audio/image)
from __future__ import annotations
import io, re, json, asyncio
from pathlib import Path
from typing import Dict, List, Tuple
from uuid import uuid4

import requests
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go

# ───────── feature probes ─────────
def _has_edge_tts() -> bool:
    try:
        import edge_tts  # noqa: F401
        return True
    except Exception:
        return False

def _has_pydub() -> bool:
    try:
        from pydub import AudioSegment  # noqa: F401
        return True
    except Exception:
        return False

EDGE_OK, PYDUB_OK = _has_edge_tts(), _has_pydub()

# ───────── paths / constants ─────────
APP_DIR     = Path(__file__).resolve().parent
MODEL_PATH  = APP_DIR / "cuisine_pipeline.joblib"
LABELS_PATH = APP_DIR / "labels.json"
ASSETS_DIR  = (APP_DIR / "assets").resolve()

TITLE = "Smart Recipe Finder"
TOPK  = 3
np.random.seed(42)

CUISINE_IMAGES = {
    "brazilian":"https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/Feijoada.jpg/800px-Feijoada.jpg",
    "british":"https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Fish_and_chips_blackpool.jpg/800px-Fish_and_chips_blackpool.jpg",
    "cajun_creole":"https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cajun_cuisine.jpg/800px-Cajun_cuisine.jpg",
    "chinese":"https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/Cuisine_of_China.jpg/800px-Cuisine_of_China.jpg",
    "filipino":"https://upload.wikimedia.org/wikipedia/commons/thumb/2/2a/Philippine_cuisine.jpg/800px-Philippine_cuisine.jpg",
    "french":"https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/French_cuisine_-_duck_confit.jpg/800px-French_cuisine_-_duck_confit.jpg",
    "greek":"https://upload.wikimedia.org/wikipedia/commons/thumb/8/81/Greek_meze.jpg/800px-Greek_meze.jpg",
    "indian":"https://upload.wikimedia.org/wikipedia/commons/thumb/0/08/Indian_cuisine.jpg/800px-Indian_cuisine.jpg",
    "irish":"https://upload.wikimedia.org/wikipedia/commons/thumb/1/1f/Irish_stew.jpg/800px-Irish_stew.jpg",
    "italian":"https://upload.wikimedia.org/wikipedia/commons/thumb/a/a3/Meal_Pizza.jpg/800px-Meal_Pizza.jpg",
    "jamaican":"https://upload.wikimedia.org/wikipedia/commons/thumb/f/f9/Jerk_chicken_%28Jamaica%29.jpg/800px-Jerk_chicken_%28Jamaica%29.jpg",
    "japanese":"https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/Sushi_platter.jpg/800px-Sushi_platter.jpg",
    "korean":"https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Korean_cuisine-Kimchi.jpg/800px-Korean_cuisine-Kimchi.jpg",
    "mexican":"https://upload.wikimedia.org/wikipedia/commons/thumb/1/10/Tacos_de_carnitas.jpg/800px-Tacos_de_carnitas.jpg",
    "moroccan":"https://upload.wikimedia.org/wikipedia/commons/thumb/2/2c/Tajine_Zitoune.jpg/800px-Tajine_Zitoune.jpg",
    "russian":"https://upload.wikimedia.org/wikipedia/commons/thumb/7/77/Borscht_served.jpg/800px-Borscht_served.jpg",
    "southern_us":"https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/Southern_US_cuisine.jpg/800px-Southern_US_cuisine.jpg",
    "spanish":"https://upload.wikimedia.org/wikipedia/commons/thumb/6/6a/Paella_mixta_01.jpg/800px-Paella_mixta_01.jpg",
    "thai":"https://upload.wikimedia.org/wikipedia/commons/thumb/0/0c/Pad_Thai_kung_Chang_Khien_street_stall.jpg/800px-Pad_Thai_kung_Chang_Khien_street_stall.jpg",
    "vietnamese":"https://upload.wikimedia.org/wikipedia/commons/thumb/5/5a/Ph%E1%BB%9F_in_Saigon.jpg/800px-Ph%E1%BB%9F_in_Saigon.jpg",
}
PLACEHOLDER = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ac/No_image_available.svg/512px-No_image_available.svg.png"

# ───────── image retrieval ─────────
@st.cache_data(show_spinner=False, ttl=3600)
def _read_local(fname: str) -> bytes | None:
    try:
        p = (ASSETS_DIR / fname).resolve()
        if not str(p).startswith(str(ASSETS_DIR)):
            return None
        if p.exists() and p.is_file():
            return p.read_bytes()
    except Exception:
        pass
    return None

@st.cache_data(show_spinner=False, ttl=3600)
def _http_bytes(url: str, timeout: int = 10) -> bytes | None:
    try:
        r = requests.get(url, timeout=timeout)
        if r.ok and r.content and "text/html" not in r.headers.get("Content-Type","").lower():
            return r.content
    except Exception:
        return None
    return None

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_image_bytes(cuisine: str) -> bytes | None:
    key = (cuisine or "").strip().lower()
    for ext in (".jpg",".jpeg",".png",".webp"):
        b = _read_local(key + ext)
        if b:
            return b
    url = CUISINE_IMAGES.get(key)
    if url:
        b = _http_bytes(url)
        if b: return b
    return _http_bytes(PLACEHOLDER)

# ───────── nutrition / heuristics ─────────
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

def parse_ingredients(text: str) -> List[str]:
    raw = [t.strip() for t in text.replace("\n", ",").split(",")]
    return [t for t in raw if t]

def parse_mass_g(s: str, default: float = 100.0) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(g|gram|grams)\b", s.lower())
    return float(m.group(1)) if m else float(default)

def estimate_nutrition(items: List[str], default_mass_g: float = 100.0) -> Dict[str,float]:
    tot = {"kcal":0.0,"protein":0.0,"fat":0.0,"carbs":0.0}
    for it in items:
        grams = parse_mass_g(it, default_mass_g)
        it_l = it.lower()
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

# ───────── model ─────────
@st.cache_resource(show_spinner="Loading model pipeline…")
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

def predict_topk(pipe, inv_labels, ings: List[str], k: int = TOPK):
    proba = pipe.predict_proba([" ".join(ings)])[0]
    order = np.argsort(proba)[::-1][:k]
    names = [inv_labels[i] for i in order]
    values = proba[order]
    return names, values

# ───────── recipes / podcast / tts ─────────
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

EDGE_FEMALE_CHOICES = ["en-US-JennyNeural","en-GB-LibbyNeural","en-AU-NatashaNeural","en-CA-ClaraNeural"]
EDGE_MALE_CHOICES   = ["en-US-GuyNeural","en-GB-RyanNeural","en-AU-WilliamNeural","en-IN-PrabhatNeural"]

def _ssml_chat(text: str, rate: str = "+0%", pitch: str = "+0%") -> str:
    safe = (text or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    return f"""<speak version="1.0" xml:lang="en-US">
  <mstts:express-as style="chat" styledegree="2" xmlns:mstts="https://www.w3.org/2001/mstts">
    <prosody rate="{rate}" pitch="{pitch}">{safe}</prosody>
  </mstts:express-as>
</speak>"""

async def _edge_async(ssml: str, voice: str) -> bytes:
    import edge_tts
    tts = edge_tts.Communicate(ssml=ssml, voice=voice)
    buf = io.BytesIO()
    async for chunk in tts.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    return buf.getvalue()

def _is_audio_bytes(x) -> bool:
    return isinstance(x, (bytes, bytearray)) and len(x) > 32

def tts_bytes_any(text: str, role: str, voice: str | None, rate: str = "+0%", pitch: str = "+0%") -> bytes | None:
    if voice and EDGE_OK:
        try:
            return asyncio.run(_edge_async(_ssml_chat(text, rate, pitch), voice))
        except Exception:
            pass
    try:
        from gtts import gTTS
        tld = "co.uk" if role == "HOST" else "com.au"
        tts = gTTS(text=text, lang="en", tld=tld)
        out = io.BytesIO(); tts.write_to_fp(out)
        return out.getvalue()
    except Exception:
        return None

def stitch_dialogue(dialogue: List[Tuple[str,str]], host_voice: str | None, chef_voice: str | None,
                    pause_ms: int = 300, rate: str = "+0%", pitch: str = "+0%") -> bytes | None:
    if not PYDUB_OK:
        return None
    try:
        from pydub import AudioSegment
        track = AudioSegment.silent(duration=50)
        gap   = AudioSegment.silent(duration=max(0, int(pause_ms)))
        for role, text in dialogue:
            v = host_voice if role == "HOST" else chef_voice
            b = tts_bytes_any(text, role, v, rate=rate, pitch=pitch)
            if not _is_audio_bytes(b): 
                return None
            seg = AudioSegment.from_file(io.BytesIO(b), format="mp3")
            track += seg + gap
        out = io.BytesIO(); track.export(out, format="mp3")
        return out.getvalue()
    except Exception:
        return None

def build_podcast_dialogue(host_name: str, chef_name: str, cuisine: str, ings: List[str],
                           nutr: Dict[str,float], tags: List[str]) -> List[Tuple[str,str]]:
    ttags = ", ".join(tags) if tags else "balanced"
    return [
        ("HOST", f"Hello everyone, I'm {host_name}, and welcome back to Flavor Talks!"),
        ("HOST", f"Today we have Chef {chef_name} with us — bringing {cuisine.title()} flavors."),
        ("CHEF", f"Hi {host_name}, thanks for having me. Our basket: {', '.join(ings)}."),
        ("HOST", f"Before we cook, what defines {cuisine.title()} cuisine in a nutshell?"),
        ("CHEF", "Balance, regional staples, and harmony — even with simple weeknight ingredients."),
        ("HOST", "Quick nutrition snapshot?"),
        ("CHEF", f"Calorie density {nutr['calorie_band']}; protein {nutr['protein_index']:.2f}; healthiness {nutr['healthiness']:.2f}. Tags: {ttags}."),
        ("HOST", "One pro tip?"),
        ("CHEF", "Bloom aromatics gently; finish with acid and herbs."),
        ("HOST", "Great — walk us through the cook."),
        ("CHEF", "Warm the pan, bloom aromatics, build core flavors, adjust, and serve."),
    ]

# ───────── UI ─────────
st.set_page_config(page_title=TITLE, page_icon="🍽️", layout="wide")
st.title(TITLE)
st.caption("Cuisine prediction • three recipe options • calories/macros • selectable voices • conversational podcast")

st.sidebar.header("Audio Settings")
st.sidebar.markdown(f"- Edge TTS available: **{'Yes' if EDGE_OK else 'No (fallback to gTTS)'}**")
st.sidebar.markdown(f"- Single MP3 stitch: **{'Yes' if PYDUB_OK else 'No (install ffmpeg)'}**")

host_voice = EDGE_FEMALE_CHOICES[0] if EDGE_OK else None
chef_voice = EDGE_MALE_CHOICES[0]   if EDGE_OK else None
if EDGE_OK:
    host_voice = st.sidebar.selectbox("Host voice (female)", EDGE_FEMALE_CHOICES, index=0, key="host_voice_sel")
    chef_voice = st.sidebar.selectbox("Chef voice (male)",   EDGE_MALE_CHOICES,   index=0, key="chef_voice_sel")
else:
    st.sidebar.info("Edge voices unavailable: gTTS fallback will be used (single voice).")

voice_rate        = st.sidebar.selectbox("Voice speed", ["-10%","-5%","+0%","+5%","+10%"], index=2, key="rate_sel")
voice_pitch       = st.sidebar.selectbox("Voice pitch", ["-2%","+0%","+2%","+4%"], index=1, key="pitch_sel")
enable_recipe_tts = st.sidebar.checkbox("Enable voice for selected recipe", value=True, key="rec_tts_chk")
enable_podcast    = st.sidebar.checkbox("Enable conversational podcast (Host ↔ Chef)", value=True, key="podcast_chk")
podcast_pause     = st.sidebar.slider("Pause between turns (ms)", 150, 800, 300, step=50, key="pause_sel")
host_name         = st.sidebar.text_input("Host display name", value="Sara", key="host_name_in")
chef_name         = st.sidebar.text_input("Chef display name", value="Masoud", key="chef_name_in")

with st.sidebar.expander("Image setup", expanded=False):
    st.markdown(
        "- Preferred: add images into `assets/` named `<cuisine>.jpg` (lowercase).\n"
        "- App also has curated Wikimedia fallbacks; otherwise a small placeholder is shown."
    )

pipe, INV = load_pipeline()
st.sidebar.markdown(f"- Classes: **{len(INV)}**")

left, right = st.columns([1.25, 1.0], vertical_alignment="top")

with left:
    st.subheader("Ingredients")
    demo = "chicken, soy sauce, ginger, garlic, sesame oil"
    ing_text = st.text_area("Comma-separated or one per line", value=demo, height=120,
                            placeholder="e.g., tomato, basil, garlic, olive oil", key="ing_text")
    ings = parse_ingredients(ing_text)

    run = st.button("Predict cuisines & build 3 recipe options", type="primary",
                    use_container_width=True, key="predict_btn")
    if run:
        if not ings:
            st.warning("Please provide at least one ingredient."); st.stop()

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
        st.session_state["selected_cuisine"] = cuisines[0]
        st.rerun()

with left:
    if st.session_state.get("pred_ready", False):
        df_pred      = st.session_state["df_pred"]
        cuisines     = st.session_state["cuisines"]
        options_meta = st.session_state["options_meta"]
        selected     = st.session_state.get("selected_cuisine", cuisines[0])

        st.markdown("### Top predictions")
        fig_pred = go.Figure(data=[go.Bar(x=df_pred["cuisine"], y=df_pred["probability"],
                                          marker_color=["#4C78A8", "#F58518", "#54A24B"])])
        fig_pred.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                               yaxis=dict(title="Proba", rangemode="tozero"),
                               xaxis=dict(title="Cuisine"), height=300, template="simple_white")
        st.plotly_chart(fig_pred, use_container_width=True, config={"displayModeBar": False}, key="pred_chart")

        st.markdown("### Explore three recipe options")
        chosen = st.radio("Pick one to preview & voice:", options=cuisines,
                          index=cuisines.index(selected), horizontal=True,
                          label_visibility="collapsed", key="recipe_radio")
        if chosen != selected:
            st.session_state["selected_cuisine"] = chosen
            selected = chosen

        c = selected
        st.markdown(f"**Cuisine:** {c.title()}")

        # image (bounded width)
        img = fetch_image_bytes(c)
        if img:
            st.image(img, width=680)
        else:
            st.info("Image unavailable for this cuisine.")

        # recipe text
        st.text_area("Recipe", value=options_meta[c]["recipe"], height=220,
                     label_visibility="collapsed", key=f"recipe_preview_{c}")

        # macros chart
        m = options_meta[c]["macro"]
        names = ["Calories (kcal)", "Protein (g)", "Fat (g)", "Carbs (g)"]
        vals  = [m["kcal"], m["protein"], m["fat"], m["carbs"]]
        fig_macro = go.Figure(data=[go.Bar(x=names, y=vals,
                                           marker_color=["#3E7CB1", "#66C2A5", "#FC8D62", "#8DA0CB"])])
        fig_macro.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                                yaxis=dict(title="Amount", rangemode="tozero"),
                                xaxis=dict(title=""), height=300, template="simple_white")
        st.plotly_chart(fig_macro, use_container_width=True, config={"displayModeBar": False}, key=f"macro_{c}")

        # value chart
        n = options_meta[c]["nutr"]
        vnames = ["Caloric density", "Protein index", "Healthiness", "Food Value Score"]
        vvals  = [n["caloric_density"], n["protein_index"], n["healthiness"], options_meta[c]["fvs"]]
        fig_val = go.Figure(data=[go.Bar(x=vnames, y=vvals,
                                         marker_color=["#A6CEE3", "#1F78B4", "#33A02C", "#FB9A99"])])
        fig_val.update_yaxes(range=[0, 1])
        fig_val.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                              yaxis=dict(title="Score (0–1)"), xaxis=dict(title=""),
                              height=300, template="simple_white")
        st.plotly_chart(fig_val, use_container_width=True, config={"displayModeBar": False}, key=f"value_{c}")

        # voice recipe (strict bytes + unique key)
        if enable_recipe_tts:
            st.markdown("#### 🔊 Voice recipe")
            audio = tts_bytes_any(options_meta[c]["recipe"], role="HOST", voice=host_voice,
                                  rate=voice_rate, pitch=voice_pitch)
            if _is_audio_bytes(audio):
                st.audio(io.BytesIO(audio), format="audio/mp3",
                         start_time=0, key=f"audio_recipe_{c}_{uuid4().hex}")
            else:
                st.info("TTS unavailable in this environment.")

        # podcast
        if enable_podcast:
            st.markdown("#### 🎙️ Conversational podcast (Host ↔ Chef)")
            tags = dietary_tags(ings)
            dlg  = build_podcast_dialogue(host_name, chef_name, c, ings, n, tags)
            st.markdown("\n".join([f"**{r}:** {t}" for r, t in dlg]))

            stitched = stitch_dialogue(dlg, host_voice, chef_voice, pause_ms=int(podcast_pause),
                                       rate=voice_rate, pitch=voice_pitch)
            if _is_audio_bytes(stitched):
                st.audio(io.BytesIO(stitched), format="audio/mp3",
                         key=f"podcast_{c}_{uuid4().hex}")
            else:
                st.caption("Per-turn playback:")
                for i, (role, text) in enumerate(dlg, 1):
                    vname = host_voice if role == "HOST" else chef_voice
                    b = tts_bytes_any(text, role, vname, rate=voice_rate, pitch=voice_pitch)
                    if not _is_audio_bytes(b):
                        st.info("TTS unavailable in this environment.")
                        break
                    st.markdown(f"*Turn {i} — {role}*")
                    st.audio(io.BytesIO(b), format="audio/mp3",
                             key=f"turn_{i}_{c}_{uuid4().hex}")

with right:
    st.subheader("How to use")
    st.markdown(
        "1) Enter ingredients\n\n"
        "2) Click **Predict cuisines & build 3 recipe options**\n\n"
        "3) Use the **radio** to pick one — preview, charts, voice & podcast all sync\n\n"
        "4) For images, add local files to `assets/` or rely on curated fallbacks\n\n"
        "5) Enable **Voice** or **Podcast** and choose voices in the sidebar"
    )

st.markdown("---")
st.caption("Meal helper • shopping assistant • cooking coach")

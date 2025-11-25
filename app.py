import json, io, os, re, math, textwrap, requests
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import joblib
from gtts import gTTS
from pydub import AudioSegment

# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = "cuisine_pipeline.joblib"
LABELS_PATH = "labels.json"

# -----------------------------
# Cache: load pipeline + labels
# -----------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_pipeline():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing {MODEL_PATH} in app root.")
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(f"Missing {LABELS_PATH} in app root.")
    pipe = joblib.load(MODEL_PATH)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)
    inv = {int(k): v for k, v in labels.items()}
    return pipe, inv

PIPE, INV = load_pipeline()
CLASSES = [INV[i] for i in sorted(INV)]

# -----------------------------
# Cleaning: same as training
# -----------------------------
def normalize_ingredient(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def join_ingredients(ings):
    return " ".join(normalize_ingredient(x) for x in ings)

# -----------------------------
# Wikipedia image fetch (safe)
# -----------------------------
WIKI_API = "https://en.wikipedia.org/w/api.php"

@st.cache_data(show_spinner=False)
def fetch_wikipedia_image(cuisine: str) -> str | None:
    """Return main image URL for a cuisine page, else None."""
    try:
        # Search page
        params = dict(
            action="query", format="json", list="search",
            srsearch=f"{cuisine} cuisine", srlimit=1
        )
        r = requests.get(WIKI_API, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data["query"]["search"]:
            return None
        title = data["query"]["search"][0]["title"]
        # Pageimage
        params2 = dict(
            action="query", format="json", prop="pageimages",
            piprop="original", titles=title
        )
        r2 = requests.get(WIKI_API, params=params2, timeout=10)
        r2.raise_for_status()
        pages = r2.json()["query"]["pages"]
        for _, pg in pages.items():
            if "original" in pg and "source" in pg["original"]:
                return pg["original"]["source"]
        return None
    except Exception:
        return None

# -----------------------------
# Simple rule-based nutrition
# -----------------------------
CAL_DB = {
    # kcal per common unit ~1 item or 100g rough averages
    "chicken": 165, "beef": 250, "pork": 242, "lamb": 258, "shrimp": 99, "fish": 206,
    "egg": 78, "milk": 60, "butter": 717, "cream": 340, "yogurt": 59, "cheese": 402,
    "rice": 130, "noodle": 138, "pasta": 131, "bread": 265, "flour": 364, "tortilla": 237,
    "potato": 77, "tomato": 18, "onion": 40, "garlic": 149, "pepper": 26, "ginger": 80,
    "oil": 884, "olive": 884, "sesame oil": 884, "ghee": 900,
    "sugar": 387, "honey": 304, "soy sauce": 60, "coconut milk": 230,
    "beans": 347, "lentil": 116, "chickpea": 164, "tofu": 76, "mushroom": 22,
    "spinach": 23, "carrot": 41, "cabbage": 25, "broccoli": 34, "eggplant": 25
}

GLUTEN_KEYS = {"wheat", "barley", "rye", "farina", "semolina", "bulgur", "bread", "flour", "pasta", "noodle", "couscous"}
NON_HALAL = {"pork", "bacon", "ham", "lard", "gelatin (porcine)", "wine", "beer", "rum", "brandy", "sake"}
ANIMAL_MEAT = {"chicken", "beef", "pork", "lamb", "goat", "turkey", "duck", "fish", "shrimp", "crab", "clam", "oyster"}
DAIRY = {"milk", "butter", "cream", "cheese", "yogurt", "ghee"}

def estimate_calories(ings: list[str], servings: int = 4) -> dict:
    tokens = [normalize_ingredient(x) for x in ings]
    total = 0.0
    for t in tokens:
        # pick best matching key in DB
        best = None
        for k in CAL_DB:
            if k in t:
                best = k; break
        if best is not None:
            total += CAL_DB[best]
        else:
            # default small add for spices/unknowns
            total += 15
    per_serv = max(total / max(servings,1), 1.0)
    # crude health flag
    oilish = any(x in " ".join(tokens) for x in ["oil", "butter", "ghee", "cream"])
    health_note = "lighter" if per_serv < 500 and not oilish else "rich"
    return {"total_kcal": int(round(total)), "per_serv_kcal": int(round(per_serv)), "note": health_note}

def diet_tags(ings: list[str]) -> list[str]:
    txt = " " + " | ".join(normalize_ingredient(x) for x in ings) + " "
    has_meat = any((" " + m + " ") in txt for m in ANIMAL_MEAT)
    has_dairy = any((" " + d + " ") in txt for d in DAIRY)
    has_non_halal = any((" " + h + " ") in txt for h in NON_HALAL)
    has_gluten = any(g in txt for g in GLUTEN_KEYS)

    tags = []
    if not has_meat and not has_dairy:
        tags.append("vegan")
    elif not has_meat and has_dairy:
        tags.append("vegetarian")
    if not has_non_halal and "pork" not in txt and "bacon" not in txt and "ham" not in txt:
        tags.append("halal-friendly")
    if not has_gluten:
        tags.append("gluten-free")
    return tags or ["general"]

# -----------------------------
# Natural-sounding recipe text
# -----------------------------
def generate_recipe_text(cuisine: str, ings: list[str]) -> str:
    base = ", ".join(ings[:6]) + ("..." if len(ings) > 6 else "")
    lines = [
        f"This is a {cuisine.replace('_',' ')} style dish built around {base}.",
        "Warm a wide pan over medium heat. Add a touch of oil and let the aromatics release their fragrance.",
        "Stir in the main ingredients and season gradually with salt, pepper, and the spices typical of this cuisine.",
        "Cook until tender but lively in texture, adding a splash of water or stock if the pan dries.",
        "Finish with fresh herbs, a squeeze of citrus, or a drizzle of sauce; taste and balance salt and acidity.",
        "Serve immediately while hot."
    ]
    return "\n\n".join(lines)

# -----------------------------
# TTS (English only here)
# -----------------------------
@st.cache_data(show_spinner=False)
def tts_bytes(text: str, lang="en", speed=1.0) -> bytes:
    tts = gTTS(text=text, lang=lang)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    audio = AudioSegment.from_file(buf, format="mp3")
    if abs(speed - 1.0) > 1e-3:
        audio = audio._spawn(audio.raw_data, overrides={"frame_rate": int(audio.frame_rate*speed)}).set_frame_rate(audio.frame_rate)
    out = io.BytesIO()
    audio.export(out, format="mp3")
    return out.getvalue()

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Smart Recipe Finder", layout="wide")

st.sidebar.header("Settings")
enable_tts = st.sidebar.checkbox("Enable Voice (English TTS)", value=True)
st.sidebar.markdown("**Model:** Logistic Regression + TF-IDF\n\n**Classes:** " + str(len(CLASSES)))

st.title("Smart Recipe Finder")
st.caption("Enter ingredients, get predicted cuisine, recipe text, image, calories, and diet tags.")

# Input
ings_input = st.text_area(
    "Ingredients (comma-separated)",
    value="chicken, soy sauce, ginger, garlic, sesame oil",
    help="e.g., tomato, basil, olive oil"
)
servings = st.number_input("Servings (for calorie estimate)", min_value=1, max_value=12, value=4, step=1)
btn = st.button("Predict & Generate", type="primary")

# Output
if btn:
    ings = [x.strip() for x in ings_input.split(",") if x.strip()]
    if not ings:
        st.warning("Please enter at least one ingredient.")
        st.stop()

    text_for_model = join_ingredients(ings)
    proba = PIPE.predict_proba([text_for_model])[0]
    order = np.argsort(proba)[::-1]
    top1_idx, top1_p = int(order[0]), float(proba[order[0]])
    top3 = [(INV[int(i)], float(proba[i])) for i in order[:3]]
    top1_cuisine = INV[top1_idx]

    # ---- chart (clean labels) ----
    df_top3 = pd.DataFrame({"Cuisine": [c for c,_ in top3], "Probability": [p for _,p in top3]})
    chart = (
        alt.Chart(df_top3)
        .mark_bar()
        .encode(
            x=alt.X("Cuisine", sort="-y"),
            y=alt.Y("Probability", scale=alt.Scale(domain=[0,1])),
            tooltip=["Cuisine", alt.Tooltip("Probability", format=".2f")]
        )
        .properties(height=260)
    )
    st.altair_chart(chart, use_container_width=True)

    st.subheader(f"Image • {top1_cuisine.replace('_',' ').title()}")
    img_url = fetch_wikipedia_image(top1_cuisine)
    if img_url:
        st.image(img_url, use_column_width=True)
    else:
        st.info("No image found on Wikipedia; showing text only.")

    # Recipe text (natural)
    st.subheader(f"🧑‍🍳 {top1_cuisine.replace('_',' ').title()} Inspired Recipe")
    recipe_text = generate_recipe_text(top1_cuisine, ings)
    st.text_area("Generated Recipe", recipe_text, height=220)

    # Calories + diet tags
    colA, colB = st.columns([1,1])
    with colA:
        est = estimate_calories(ings, servings=servings)
        st.metric("Estimated kcal (total)", est["total_kcal"])
        st.metric("Estimated kcal / serving", est["per_serv_kcal"])
        st.caption(f"Profile: **{est['note']}** (very rough, heuristic).")
    with colB:
        tags = diet_tags(ings)
        st.markdown("**Diet tags:** " + " • ".join(f"`{t}`" for t in tags))

    # Optional voice
    if enable_tts:
        audio_bytes = tts_bytes(recipe_text, lang="en", speed=1.0)
        st.audio(audio_bytes, format="audio/mp3")

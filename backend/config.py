# ============================================================
# config.py — Centralized configuration for the Fashion AI system
# ============================================================

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Root paths ───────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent.parent          # ML-TASK-main/
DATA_DIR   = ROOT_DIR / "data"
IMAGE_DIR  = DATA_DIR / "images"
VECTOR_DIR = ROOT_DIR / "vector_db"
MODEL_DIR  = ROOT_DIR / "models"

# ── Data files ───────────────────────────────────────────────
PRODUCTS_CSV  = DATA_DIR / "products.csv"
OUTFITS_CSV   = DATA_DIR / "outfits.csv"
CURATED_XLSX  = DATA_DIR / "curated25.xlsx"

# ── API keys ─────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_TEMPERATURE = 0.7
GEMINI_MAX_TOKENS = 800

# ── Embedding / CLIP model ───────────────────────────────────
CLIP_MODEL_NAME   = "openai/clip-vit-base-patch32"   # HuggingFace CLIP
EMBED_DIM         = 512
TEXT_WEIGHT       = 0.40   # weight for text embedding in fusion
IMAGE_WEIGHT      = 0.60   # weight for image embedding in fusion

# ── Persisted artifact paths ─────────────────────────────────
FAISS_INDEX_PATH    = VECTOR_DIR / "fashion.index"
PRODUCT_META_PATH   = VECTOR_DIR / "product_meta.pkl"
EMBEDDINGS_NPY_PATH = MODEL_DIR  / "embeddings.npy"

# ── Recommendation settings ──────────────────────────────────
TOP_K_CANDIDATES = 10   # candidates per slot from FAISS
TOP_K_OUTFITS    = 3    # final outfit count to return
SIMILARITY_THRESHOLD = 0.15

# ── Category → Outfit slot mapping ───────────────────────────
TOPWEAR_CATEGORIES = {
    "formal-shirts", "casual-shirts", "t-shirts", "shirts",
    "sweatshirts", "hoodies", "sweaters", "blouses", "tops",
    "crop-tops", "kurtas", "jackets", "blazers", "suits", "dresses",
    "kurta-sets", "co-ord-sets", "jumpsuits",
}
BOTTOMWEAR_CATEGORIES = {
    "trousers", "jeans", "chinos", "cargo-pants", "shorts",
    "skirts", "leggings", "palazzos", "track-pants",
}
FOOTWEAR_CATEGORIES = {
    "sneakers", "formal-shoes", "loafers", "sandals", "heels",
    "boots", "flats", "sports-shoes", "shoes", "footwear",
    "pumps", "slippers", "mules",
}
ACCESSORY_CATEGORIES = {
    "watches", "belts", "bags", "handbags", "clutches",
    "jewellery", "sunglasses", "caps", "hats", "wallets",
    "backpacks", "sling-bags", "scarves",
}

SLOT_MAP = {
    "topwear":    TOPWEAR_CATEGORIES,
    "bottomwear": BOTTOMWEAR_CATEGORIES,
    "footwear":   FOOTWEAR_CATEGORIES,
    "accessory":  ACCESSORY_CATEGORIES,
}

# ── Occasion compatibility ────────────────────────────────────
OCCASION_GROUPS = {
    "formal":   {"office", "interview", "formal", "business"},
    "casual":   {"casual", "everyday", "daily", "street"},
    "party":    {"party", "evening", "night-out", "cocktail"},
    "wedding":  {"wedding", "festive", "occasion"},
    "outdoor":  {"outdoor", "sports", "gym", "beach", "travel", "vacation"},
}

# ── Color palette knowledge ───────────────────────────────────
NEUTRAL_COLORS  = {"white", "black", "grey", "beige", "cream", "navy", "brown"}
WARM_COLORS     = {"red", "orange", "yellow", "pink", "maroon", "burgundy", "rust"}
COOL_COLORS     = {"blue", "green", "purple", "teal", "mint", "lavender"}

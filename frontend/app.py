# ============================================================
# app.py — Streamlit Fashion Assistant UI
# ============================================================

import streamlit as st
import sys
from pathlib import Path

# Ensure backend is importable when running from frontend/
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import time
import numpy as np
from PIL import Image

from backend.config import IMAGE_DIR, DATA_DIR
from backend.data_loader import DataLoader
from backend.recommendation.recommendation_engine import RecommendationEngine
from backend.chatbot.gemini_chatbot import GeminiFashionChatbot

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Anku AI· Fashion Assistant",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@400;600;700&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #F5F5F5;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Playfair Display', serif !important;
    font-weight: 600;
}

/* ── Dark background ── */
.stApp {
    background: #0F0F0F;
    color: #F5F5F5;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #181818 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

[data-testid="stSidebar"] * { color: #B0B0B0 !important; }

/* ── Header ── */
.app-header {
    text-align: center;
    padding: 3rem 0 1rem 0;
    margin-bottom: 2rem;
}
.app-header h1 {
    font-size: 3rem;
    color: #F5F5F5;
    letter-spacing: 0.1em;
    margin: 0;
    font-weight: 700;
}
.app-header p {
    color: #B0B0B0;
    font-size: 1.1rem;
    margin-top: 0.5rem;
    letter-spacing: 0.05em;
    font-weight: 300;
}

/* ── Hero Banner ── */
.hero-banner {
    text-align: center;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    background: #181818;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}
.hero-banner h2 {
    font-size: 2.5rem;
    color: #C9A227;
    margin-bottom: 1rem;
    font-weight: 600;
}
.hero-banner p {
    color: #B0B0B0;
    font-size: 1.1rem;
    max-width: 600px;
    margin: 0 auto;
    line-height: 1.6;
    font-weight: 300;
}

/* ── Chat container ── */
.chat-container {
    background: rgba(24, 24, 24, 0.6);
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    padding: 2rem;
    margin-bottom: 2rem;
    backdrop-filter: blur(20px);
}

/* ── User/assistant bubbles ── */
.user-bubble {
    background: #181818;
    border: 1px solid rgba(201, 162, 39, 0.3);
    border-radius: 12px 12px 4px 12px;
    padding: 1rem 1.5rem;
    margin: 0.8rem 0;
    margin-left: 15%;
    color: #F5F5F5;
    font-size: 0.95rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}
.assistant-bubble {
    background: rgba(24, 24, 24, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px 12px 12px 4px;
    padding: 1rem 1.5rem;
    margin: 0.8rem 0;
    margin-right: 15%;
    color: #F5F5F5;
    font-size: 0.95rem;
    backdrop-filter: blur(10px);
}

/* ── Cards (Product & Outfit) ── */
.premium-card {
    background: rgba(24, 24, 24, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 0.8rem 0;
    transition: all 0.3s ease;
    backdrop-filter: blur(20px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    position: relative;
    overflow: hidden;
}
.premium-card:hover {
    border-color: rgba(201, 162, 39, 0.4);
    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    transform: translateY(-6px);
}

/* ── Card Badges ── */
.badge {
    position: absolute;
    top: 1rem;
    right: 1rem;
    padding: 0.3rem 0.8rem;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    z-index: 10;
}
.badge-premium { background: #C9A227; color: #0F0F0F; }
.badge-value { background: #2ECC71; color: #0F0F0F; }
.badge-trending { background: #F5F5F5; color: #0F0F0F; }

/* ── Typography inside cards ── */
.card-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    color: #F5F5F5;
    margin-bottom: 0.5rem;
    margin-top: 0;
}
.card-price {
    color: #C9A227;
    font-weight: 600;
    font-size: 1.1rem;
    margin-bottom: 0.8rem;
}
.card-match {
    color: #2ECC71;
    font-size: 0.85rem;
    font-weight: 500;
    margin-bottom: 0.5rem;
}
.card-tags {
    color: #B0B0B0;
    font-size: 0.8rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.card-desc {
    color: #B0B0B0;
    font-size: 0.9rem;
    line-height: 1.5;
    font-style: italic;
    border-left: 2px solid #C9A227;
    padding-left: 1rem;
    margin-top: 1rem;
}

/* ── Input styling ── */
.stTextInput > div > div > input {
    background: #181818 !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 8px !important;
    color: #F5F5F5 !important;
    padding: 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #C9A227 !important;
    box-shadow: 0 0 0 1px rgba(201, 162, 39, 0.5) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #C9A227 !important;
    color: #0F0F0F !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    transition: all 0.3s ease !important;
    padding: 0.5rem 2rem !important;
}
.stButton > button:hover {
    background: #F5F5F5 !important;
    color: #0F0F0F !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 15px rgba(255, 255, 255, 0.1) !important;
}

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    padding: 0;
    margin-bottom: 2rem;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #B0B0B0;
    border-radius: 0;
    border: none;
    padding: 1rem 2rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-size: 0.85rem;
}
.stTabs [aria-selected="true"] {
    color: #C9A227 !important;
    background: transparent !important;
    border-bottom: 2px solid #C9A227 !important;
}

/* ── Divider ── */
hr { border-color: rgba(255, 255, 255, 0.05) !important; margin: 3rem 0; }

.luxury-divider {
    width: 120px;
    height: 2px;
    background: #C9A227;
    margin: 1.5rem auto 0 auto;
}
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "engine" not in st.session_state:
        st.session_state.engine = None
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = None
    if "loader" not in st.session_state:
        st.session_state.loader = None
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    if "last_outfits" not in st.session_state:
        st.session_state.last_outfits = []
    if "last_products" not in st.session_state:
        st.session_state.last_products = []
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = {}


# ── Cached initialisation ─────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_system():
    """Load all ML components once (cached across reruns)."""
    loader  = DataLoader()
    engine  = RecommendationEngine()
    chatbot = GeminiFashionChatbot()
    engine.initialize()
    chatbot.initialize()
    return loader, engine, chatbot


# ── Image helper ─────────────────────────────────────────────
def load_product_image(image_rel_path: str, loader: DataLoader):
    """Load a product image, return PIL Image or None."""
    if not image_rel_path:
        return None
    path = loader.get_image_path(image_rel_path)
    if path and path.exists():
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return None
    return None


# ── Outfit rendering ─────────────────────────────────────────
def render_outfit_card(outfit: dict, outfit_num: int, loader: DataLoader):
    """Render a single outfit recommendation card."""
    items = outfit.get("items", [])
    score = outfit.get("score", 0)
    explanation = outfit.get("explanation", "")

    # Dynamic title based on occasion
    occasions = [i.get("occasion", "").lower() for i in items if i.get("occasion")]
    dominant_occ = max(set(occasions), key=occasions.count) if occasions else "casual"
    
    title_map = {
        "party": "Elegant Evening",
        "office": "Office Essential",
        "wedding": "Wedding Guest Look",
        "casual": "Weekend Casual",
        "sports": "Activewear Edit",
        "vacation": "Resort Ready",
        "festive": "Festive Collection",
        "winter": "Winter Layering"
    }
    title = title_map.get(dominant_occ, "Curated Edit")

    with st.container():
        cols = st.columns(min(len(items), 4))
        total_price = 0
        
        images_html = []
        details_html = []

        for i, item in enumerate(items[:4]):
            img = load_product_image(item.get("image", ""), loader)
            if img:
                with cols[i]:
                    st.image(img, use_container_width=True)
            
            slot = item.get("slot", "item").title()
            name = item.get("name", "")[:40]
            price = int(item.get("price_inr", 0))
            cat_label = item.get("category_label", "")
            total_price += price

            details_html.append(f"""<div style="flex: 1; padding: 0 0.5rem;">
<div style="font-size: 0.7rem; color: #8892a4; text-transform: uppercase;">{slot}</div>
<div style="font-weight: 500; font-size: 0.85rem; margin: 0.2rem 0; color: #F5F5F5;">{name}</div>
<div style="color: #C9A227; font-weight: 600; font-size: 0.9rem;">₹{price:,}</div>
</div>""")

        st.markdown(f"""
<div class="premium-card">
<h3 class="card-title">{title}</h3>
<div class="card-match">{score*100:.0f}% Match</div>
<div class="card-tags">MODERN • SOPHISTICATED • PREMIUM</div>
<div style="display: flex; justify-content: space-between; margin-top: 1rem; margin-bottom: 1rem;">
{''.join(details_html)}
</div>
<div style="margin-top:1rem; border-top:1px solid rgba(255,255,255,0.05); padding-top:1rem;">
<div style="color:#B0B0B0; font-size:0.9rem; margin-bottom: 0.5rem;">
Total Outfit Price: <span style="color:#F5F5F5; font-weight:600">₹{total_price:,}</span>
</div>
<div class="card-desc">{explanation}</div>
</div>
</div>
""", unsafe_allow_html=True)


def render_product_grid(products: list, loader):
    """Render a grid of single products with premium styling."""
    for idx, row in enumerate(products):
        price = int(row.get("price_inr", 0))
        score = row.get("score", 0.90) # Dummy score if unavailable for single items
        
        badge_html = ""
        if price > 4000:
            badge_html = "<div class='badge badge-premium'>Premium Pick</div>"
        elif price < 1500:
            badge_html = "<div class='badge badge-value'>Best Value</div>"
        else:
            badge_html = "<div class='badge badge-trending'>Recommended</div>"

        c1, c2 = st.columns([1, 4])
        with c1:
            img = load_product_image(row.get("image", ""), loader)
            if img:
                st.image(img, use_container_width=True)
            else:
                st.markdown("<div style='height: 150px; background: rgba(255,255,255,0.05); border-radius: 8px;'></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
<div class="premium-card" style="margin-top: 0;">
{badge_html}
<h3 class="card-title" style="font-size: 1.2rem;">{row.get('name', 'Unknown')}</h3>
<div class="card-match">{score*100:.0f}% Match</div>
<div class="card-price">₹{price:,}</div>
<div class="card-tags">{row.get('category_label', 'Fashion')}</div>
<div class="card-desc" style="border: none; padding-left: 0; margin-top: 0.5rem;">{row.get('description', '')[:150]}...</div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:1rem 0">
            <div style="font-size:2rem;font-family:'Playfair Display', serif;font-weight:700;color:#F5F5F5;letter-spacing:0.1em;line-height:1;">AnKu AI</div>
            <div style="font-size:0.75rem;color:#C9A227;margin-top:0.5rem;text-transform:uppercase;letter-spacing:0.1em;">Personal Stylist</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        
        # Check Gemini status (silently handled)
        import os
        from backend.config import GEMINI_API_KEY
        try:
            import google.generativeai
            gemini_available = True
        except ImportError:
            gemini_available = False
            
        st.markdown("**👤 Your Profile**")

        gender = st.selectbox(
            "Gender", ["— Select —", "men", "women"],
            key="gender_select"
        )
        age = st.slider("Age", 16, 65, 24, key="age_slider")
        occasion = st.selectbox(
            "Occasion",
            ["— Select —", "casual", "office", "wedding", "party", "date", "beach", "interview"],
            key="occasion_select"
        )
        style = st.selectbox(
            "Style Preference",
            ["— Select —", "formal", "smart-casual", "casual", "streetwear", "sporty"],
            key="style_select"
        )
        max_price = st.slider("Max Budget (₹)", 500, 15000, 5000, step=500, key="price_slider")

        st.divider()
        if st.button("🔄 Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_outfits = []
            if st.session_state.chatbot:
                st.session_state.chatbot.reset_conversation()
            st.rerun()

        st.divider()
        st.markdown("**💡 Try asking:**")
        examples = [
            "Need an outfit for a business meeting",
            "Casual summer outfit for a 22-year-old",
            "Smart casual look for a dinner date",
            "Outfit for a beach vacation",
            "Wedding guest attire",
            "I need shoes under 1000 for men"
        ]
        for ex in examples:
            if st.button(f"→ {ex}", use_container_width=True, key=f"ex_{ex[:20]}"):
                st.session_state.pending_message = ex
                st.rerun()

        # Profile object
        profile = {}
        if gender != "— Select —":
            profile["gender"] = gender
        profile["age"] = age
        if occasion != "— Select —":
            profile["occasion"] = occasion
        if style != "— Select —":
            profile["style"] = style
        profile["max_price"] = max_price
        st.session_state.user_profile = profile

        return profile


# ── Main app ─────────────────────────────────────────────────
def main():
    init_session_state()

    # ── Header & Hero ─────────────────────────────────────────
    st.markdown("""
    <div class="app-header">
        <h1>AnKu AI</h1>
        <p>Curated Fashion Recommendations Tailored To Your Style</p>
        <div class="luxury-divider"></div>
    </div>
    
    <div class="hero-banner">
        <h2>Find Your Signature Style</h2>
        <p>Discover curated outfits and fashion recommendations designed around your personality, occasion, and budget.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar (profile) ─────────────────────────────────────
    profile = render_sidebar()

    # ── Load system ───────────────────────────────────────────
    if not st.session_state.initialized:
        with st.spinner("🚀 Loading AI models (first run may take ~60s) …"):
            try:
                loader, engine, chatbot = load_system()
                st.session_state.loader  = loader
                st.session_state.engine  = engine
                st.session_state.chatbot = chatbot
                st.session_state.initialized = True
            except Exception as e:
                st.error(f"System initialization failed: {e}")
                st.info("Please ensure all requirements are installed and the dataset is in `data/`.")
                return

    loader  = st.session_state.loader
    engine  = st.session_state.engine
    chatbot = st.session_state.chatbot

    # ── Tabs ─────────────────────────────────────────────────
    tab_chat, tab_explore, tab_stats = st.tabs(
        ["Personal Stylist", "Explore Collection", "Collection Overview"]
    )

    # ─────────────────────────────────────────────────────────
    # TAB 1: Chat
    # ─────────────────────────────────────────────────────────
    with tab_chat:
        # Welcome message
        if not st.session_state.messages:
            welcome = (
                "Welcome to AnKu AI.\n\n"
                "Tell us where you're going, your preferred style, or a fashion item you have in mind, "
                "and we'll curate recommendations tailored to you."
            )
            st.session_state.messages.append({"role": "assistant", "content": welcome})

        # Chat history
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="user-bubble">🙋 {msg["content"]}</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-bubble">🤖 {msg["content"]}</div>',
                            unsafe_allow_html=True)

        # Outfit cards below latest recommendation
        if st.session_state.last_outfits:
            st.markdown("---")
            if "fallback_outfits" in st.session_state and st.session_state.fallback_outfits:
                st.markdown(f"### ✨ We found {len(st.session_state.last_outfits)} exact match{'es' if len(st.session_state.last_outfits) != 1 else ''} within your budget.")
                for idx, outfit in enumerate(st.session_state.last_outfits, 1):
                    render_outfit_card(outfit, idx, loader)
                st.markdown("### 🔍 To give you more options, we've also included similar outfits slightly above your budget.")
                for idx, outfit in enumerate(st.session_state.fallback_outfits, 1):
                    render_outfit_card(outfit, idx, loader)
            else:
                st.markdown("### ✨ Recommended Outfits")
                for idx, outfit in enumerate(st.session_state.last_outfits, 1):
                    render_outfit_card(outfit, idx, loader)
                
        # Product cards below latest recommendation
        if st.session_state.last_products:
            st.markdown("---")
            if "fallback_products" in st.session_state and st.session_state.fallback_products:
                st.markdown(f"### ✨ We found {len(st.session_state.last_products)} exact match{'es' if len(st.session_state.last_products) != 1 else ''} within your budget.")
                render_product_grid(st.session_state.last_products, loader)
                st.markdown("### 🔍 To give you more options, we've also included similar products slightly above your budget.")
                render_product_grid(st.session_state.fallback_products, loader)
            else:
                render_product_grid(st.session_state.last_products, loader)

        # Chat input logic
        user_input = st.chat_input("What are you dressing for today?")
        
        msg = None
        if user_input:
            msg = user_input.strip()
        elif "pending_message" in st.session_state and st.session_state.pending_message:
            msg = st.session_state.pop("pending_message").strip()

        if msg:
            st.session_state.messages.append({"role": "user", "content": msg})

            with st.spinner("Styling your outfit ..."):
                # Step 1: Extract intent
                intent = chatbot.extract_intent(msg)

                # Prioritise intent over sidebar profile
                gender  = intent.get("gender") or profile.get("gender")
                age     = intent.get("age")    or profile.get("age")
                occasion= intent.get("occasion") or profile.get("occasion")
                style   = intent.get("style")    or profile.get("style")
                max_p   = intent.get("max_price_inr") or profile.get("max_price")

                query_intent = intent.get("query_intent", "complete_outfit")
                items_mentioned = intent.get("items_mentioned", [])
                
                if query_intent == "find_item" or items_mentioned:
                    # PRODUCT SEARCH MODE
                    product_type = items_mentioned[0] if items_mentioned else None
                    products = engine.search_products(
                        query_text=intent.get("enriched_query", msg),
                        product_type=product_type,
                        gender=gender,
                        max_price=max_p,
                        top_k=4
                    )
                    
                    st.session_state.fallback_products = []
                    if len(products) < 3 and max_p is not None:
                        fallback = engine.search_products(
                            query_text=intent.get("enriched_query", msg),
                            product_type=product_type,
                            gender=gender,
                            max_price=None,
                            top_k=4
                        )
                        exact_ids = {p["id"] for p in products}
                        st.session_state.fallback_products = [p for p in fallback if p["id"] not in exact_ids]

                    st.session_state.last_outfits = []
                    st.session_state.fallback_outfits = []
                    st.session_state.last_products = products
                    
                    # Context for chat response
                    context_items = products
                    if not products and st.session_state.fallback_products:
                        context_items = st.session_state.fallback_products
                    reply = chatbot.chat(msg, outfit_context=context_items)
                else:
                    # OUTFIT MODE
                    outfits = engine.recommend_from_profile(
                        query_text=intent.get("enriched_query", msg),
                        gender=gender,
                        age=age,
                        occasion=occasion,
                        style=style,
                        max_price=max_p,
                        top_k=3,
                    )
                    
                    st.session_state.fallback_outfits = []
                    if len(outfits) < 3 and max_p is not None:
                        fallback_outfits = engine.recommend_from_profile(
                            query_text=intent.get("enriched_query", msg),
                            gender=gender,
                            age=age,
                            occasion=occasion,
                            style=style,
                            max_price=None,
                            top_k=3,
                        )
                        exact_ids = []
                        for o in outfits:
                            exact_ids.append({i["id"] for i in o["items"]})
                        
                        fallback_deduped = []
                        for fo in fallback_outfits:
                            fo_ids = {i["id"] for i in fo["items"]}
                            if fo_ids not in exact_ids:
                                fallback_deduped.append(fo)
                        st.session_state.fallback_outfits = fallback_deduped
                    
                    # Generate explanations for all
                    for outfit in outfits:
                        outfit["explanation"] = chatbot.generate_outfit_explanation(
                            outfit["items"], occasion=occasion, user_context=msg
                        )
                    for outfit in st.session_state.fallback_outfits:
                        outfit["explanation"] = chatbot.generate_outfit_explanation(
                            outfit["items"], occasion=occasion, user_context=msg
                        )

                    st.session_state.last_outfits = outfits
                    st.session_state.last_products = []
                    st.session_state.fallback_products = []
                    
                    
                    outfit_context = None
                    if outfits:
                        outfit_context = outfits[0]["items"]
                    elif st.session_state.fallback_outfits:
                        outfit_context = st.session_state.fallback_outfits[0]["items"]
                        
                    reply = chatbot.chat(msg, outfit_context=outfit_context)

                # --- DEBUG LOGGING ---
                print("\n=== DEBUG LOGS ===")
                print(f"User Query: {msg}")
                print(f"Extracted Intent: {intent}")
                print(f"Route Selected: {'PRODUCT SEARCH MODE' if (query_intent == 'find_item' or items_mentioned) else 'OUTFIT MODE'}")
                print(f"Products Found (Exact): {len(st.session_state.last_products)}")
                print(f"Fallback Products Found: {len(st.session_state.get('fallback_products', []))}")
                print(f"Outfits Found (Exact): {len(st.session_state.last_outfits)}")
                print(f"Fallback Outfits Found: {len(st.session_state.get('fallback_outfits', []))}")
                print(f"SessionState last_products length: {len(st.session_state.last_products)}")
                print(f"SessionState last_outfits length: {len(st.session_state.last_outfits)}")
                print("==================\n")

                st.session_state.messages.append({"role": "assistant", "content": reply})

            st.rerun()

    # ─────────────────────────────────────────────────────────
    # TAB 2: Explore Products
    # ─────────────────────────────────────────────────────────
    with tab_explore:
        st.markdown("### 🔍 Explore & Search Products")
        col1, col2, col3 = st.columns(3)
        with col1:
            search_q = st.text_input("Search products", placeholder="e.g. blue jeans, formal shirt...")
        with col2:
            filter_slot = st.selectbox(
                "Category slot", ["All", "topwear", "bottomwear", "footwear", "accessory"]
            )
        with col3:
            filter_gender = st.selectbox("Gender", ["All", "men", "women"])

        if st.button("🔍 Search", key="search_btn"):
            products = loader.load_products()
            filtered = products.copy()
            if filter_slot != "All":
                filtered = filtered[filtered["slot"] == filter_slot]
            if filter_gender != "All":
                filtered = filtered[filtered["gender"].str.lower() == filter_gender]
            if search_q:
                q = search_q.lower()
                mask = (
                    filtered["name"].str.lower().str.contains(q, na=False) |
                    filtered["description"].str.lower().str.contains(q, na=False) |
                    filtered["category_label"].str.lower().str.contains(q, na=False)
                )
                filtered = filtered[mask]

            if filtered.empty:
                st.warning("No products found matching your criteria.")
            else:
                st.markdown(f"**{len(filtered)} products found**")
                for _, row in filtered.iterrows():
                    c1, c2 = st.columns([1, 4])
                    with c1:
                        img = load_product_image(row.get("image", ""), loader)
                        if img:
                            st.image(img, use_container_width=True)
                        else:
                            st.markdown("👕")
                    with c2:
                        st.markdown(f"**{row['name']}**")
                        st.markdown(f"*{row.get('brand','')}* · {row.get('category_label','')} · ₹{int(row.get('price_inr',0)):,}")
                        st.markdown(f"🏷️ {row.get('occasion','')} · {row.get('gender','')}")
                        desc = row.get("description", "")
                        if desc:
                            st.caption(desc[:120] + ("…" if len(desc) > 120 else ""))
                        # Quick outfit button
                        if st.button(f"👗 Build outfit with this", key=f"outfit_btn_{row['id']}"):
                            outfits = engine.recommend_from_item(
                                row["id"],
                                gender=row.get("gender"),
                                occasion=row.get("occasion"),
                                top_k=2,
                            )
                            st.session_state.last_outfits = outfits
                            st.session_state.messages.append({
                                "role": "user",
                                "content": f"Build an outfit with: {row['name']}"
                            })
                            reply = chatbot.chat(
                                f"Build an outfit with: {row['name']}",
                                outfit_context=outfits[0]["items"] if outfits else None,
                            )
                            st.session_state.messages.append({"role": "assistant", "content": reply})
                            st.rerun()
                    st.divider()

    # ─────────────────────────────────────────────────────────
    # TAB 3: Collection Overview
    # ─────────────────────────────────────────────────────────
    with tab_stats:
        st.markdown("### 📊 Collection Overview")
        stats = loader.get_dataset_stats()

        col1, col2, col3, col4 = st.columns(4)
        metrics = [
            (stats["total_products"], "Total Products"),
            (stats["total_outfits"],  "Curated Outfits"),
            (stats["images_available"], "Product Images"),
            (len(stats["category_counts"]), "Categories"),
        ]
        for col, (val, label) in zip([col1, col2, col3, col4], metrics):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{val}</div>
                    <div class="metric-label">{label}</div>
                </div>""", unsafe_allow_html=True)

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Gender Distribution**")
            import pandas as pd
            gender_df = pd.DataFrame.from_dict(
                stats["gender_counts"], orient="index", columns=["count"]
            )
            st.bar_chart(gender_df)

            st.markdown("**Outfit Slot Distribution**")
            slot_df = pd.DataFrame.from_dict(
                stats["slot_counts"], orient="index", columns=["count"]
            )
            st.bar_chart(slot_df)

        with c2:
            st.markdown("**Occasion Distribution**")
            occ_df = pd.DataFrame.from_dict(
                stats["occasion_counts"], orient="index", columns=["count"]
            )
            st.bar_chart(occ_df)

            st.markdown("**Source Site Distribution**")
            site_df = pd.DataFrame.from_dict(
                stats["site_counts"], orient="index", columns=["count"]
            )
            st.bar_chart(site_df)

        st.divider()
        st.markdown(f"**Average Price:** ₹{stats['avg_price_inr']:,.0f}")


if __name__ == "__main__":
    main()

import sys
import os
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.chatbot.gemini_chatbot import GeminiFashionChatbot
from backend.recommendation.recommendation_engine import RecommendationEngine
from backend.config import FOOTWEAR_CATEGORIES, TOPWEAR_CATEGORIES, BOTTOMWEAR_CATEGORIES, ACCESSORY_CATEGORIES

def run_trace():
    sys.stdout.reconfigure(encoding='utf-8')
    print(f"==================================================")
    print(f"VERIFY A. FOOTWEAR_CATEGORIES: {FOOTWEAR_CATEGORIES}")
    print(f"==================================================")

    engine = RecommendationEngine()
    chatbot = GeminiFashionChatbot()
    engine.initialize()
    chatbot.initialize()
    
    query = "i need shoes under 1000 for men"
    
    # 1. extract_intent
    print(f"==================================================")
    print(f"STEP: extract_intent")
    print(f"INPUT: {query}")
    intent = chatbot.extract_intent(query)
    print(f"OUTPUT: {intent}")
    
    query_intent = intent.get("query_intent")
    items_mentioned = intent.get("items_mentioned", [])
    gender = intent.get("gender")
    max_price = intent.get("max_price_inr")
    enriched_query = intent.get("enriched_query", query)
    
    # 2. app.py routing
    print(f"==================================================")
    print(f"STEP: app.py routing")
    print(f"INPUT: query_intent={query_intent}, items_mentioned={items_mentioned}")
    if query_intent == "find_item" or items_mentioned:
        route = "PRODUCT SEARCH MODE"
        product_type = items_mentioned[0] if items_mentioned else None
    else:
        route = "OUTFIT MODE"
        product_type = None
    print(f"OUTPUT: route={route}, product_type={product_type}")

    print(f"==================================================")
    print(f"VERIFY B. product_type received: {product_type}")
    print(f"==================================================")
    
    if route == "PRODUCT SEARCH MODE":
        # 3. search_products
        print(f"==================================================")
        print(f"STEP: search_products")
        print(f"INPUT: product_type={product_type}, gender={gender}, max_price={max_price}, top_k=4")
        
        filters = {}
        if gender: filters["gender"] = gender
        if max_price is not None: filters["max_price"] = max_price
        
        # Slot mapping
        print(f"==================================================")
        print(f"STEP: slot mapping (inside search_products)")
        print(f"INPUT: product_type={product_type}")
        if product_type:
            pt = product_type.lower()
            if pt in ("topwear", "bottomwear", "footwear", "accessory"):
                filters["slot"] = pt
            elif any(k in pt for k in TOPWEAR_CATEGORIES):
                filters["slot"] = "topwear"
            elif any(k in pt for k in BOTTOMWEAR_CATEGORIES):
                filters["slot"] = "bottomwear"
            elif any(k in pt for k in FOOTWEAR_CATEGORIES):
                filters["slot"] = "footwear"
            elif any(k in pt for k in ACCESSORY_CATEGORIES):
                filters["slot"] = "accessory"
        print(f"OUTPUT: filters={filters}")
        print(f"==================================================")
        print(f"VERIFY C. Whether slot='footwear' is applied: {'slot' in filters and filters['slot'] == 'footwear'}")
        print(f"==================================================")
        
        # 4. search_by_text
        print(f"==================================================")
        print(f"STEP: search_by_text")
        print(f"INPUT: query_text={enriched_query}, filters={filters}, top_k=4")
        
        candidate_indices = engine.faiss._apply_filters(filters)
        print(f"==================================================")
        print(f"STEP: _apply_filters")
        print(f"INPUT: filters={filters}")
        print(f"OUTPUT: {len(candidate_indices)} candidate items passed filters")
        
        print(f"==================================================")
        print(f"VERIFY D. How many products remain after filtering: {len(candidate_indices)}")
        print(f"==================================================")
        
        all_meta = engine.faiss.meta_list
        print(f"Total products in DB: {len(all_meta)}")
        
        pass_gender = 0
        pass_price = 0
        pass_slot = 0
        
        for meta in all_meta:
            g = meta.get("gender", "").lower()
            if g in ("men", "unisex", ""): pass_gender += 1
            p = float(meta.get("price_inr", 0))
            if p <= 1000: pass_price += 1
            s = meta.get("slot", "")
            if s == "footwear": pass_slot += 1
            
        print(f"==================================================")
        print(f"VERIFY E. Whether gender filter removes all products: {pass_gender} pass gender 'men' or 'unisex'")
        print(f"VERIFY F. Whether price filter removes all products: {pass_price} pass max_price=1000")
        print(f"Also slot filter: {pass_slot} pass slot='footwear'")
        
        products = engine.faiss.search_by_text(enriched_query, top_k=4, filters=filters)
        print(f"==================================================")
        print(f"STEP: FAISS retrieval")
        print(f"OUTPUT: {len(products)} products returned")
        for p in products:
            print(f"  -> {p['name']} | Slot: {p.get('slot')} | Price: {p.get('price_inr')} | Gender: {p.get('gender')}")
        
    else:
        print("Test failed: Did not enter PRODUCT SEARCH MODE")

if __name__ == "__main__":
    run_trace()

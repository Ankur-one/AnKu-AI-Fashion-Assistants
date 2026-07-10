import sys
import os
from pathlib import Path

# Setup path so we can import backend
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.chatbot.gemini_chatbot import GeminiFashionChatbot
from backend.recommendation.recommendation_engine import RecommendationEngine

def run_tests():
    sys.stdout.reconfigure(encoding='utf-8')
    print("Initializing engine and chatbot...")
    engine = RecommendationEngine()
    chatbot = GeminiFashionChatbot()
    engine.initialize()
    chatbot.initialize()
    
    queries = [
        "I need shoes under 1000 for men",
        "Wedding outfit for men",
        "Casual outfit under 3000"
    ]
    
    for query in queries:
        print(f"\n=======================================================")
        print(f"QUERY: '{query}'")
        print(f"=======================================================")
        
        intent = chatbot.extract_intent(query)
        print("Intent Extracted:", intent)
        
        query_intent = intent.get("query_intent")
        items_mentioned = intent.get("items_mentioned", [])
        
        if query_intent == "find_item" or items_mentioned:
            print("\nMODE: Product Search")
            product_type = items_mentioned[0] if items_mentioned else None
            products = engine.search_products(
                query_text=intent.get("enriched_query", query),
                product_type=product_type,
                gender=intent.get("gender"),
                max_price=intent.get("max_price_inr"),
                top_k=4
            )
            for p in products:
                print(f"  -> {p['name']} | Slot: {p.get('slot')} | Price: {p.get('price_inr')} | Gender: {p.get('gender')}")
        else:
            print("\nMODE: Outfit Search")
            outfits = engine.recommend_from_profile(
                query_text=intent.get("enriched_query", query),
                gender=intent.get("gender"),
                occasion=intent.get("occasion"),
                max_price=intent.get("max_price_inr"),
                top_k=2
            )
            for i, o in enumerate(outfits):
                print(f"\nOutfit {i+1} (Score: {o['score']})")
                total_price = sum(item.get("price_inr", 0) for item in o["items"])
                for p in o["items"]:
                    print(f"  -> {p['name']} | Slot: {p.get('slot')} | Price: {p.get('price_inr')} | Gender: {p.get('gender')}")
                print(f"  Total Price: {total_price}")

if __name__ == "__main__":
    run_tests()

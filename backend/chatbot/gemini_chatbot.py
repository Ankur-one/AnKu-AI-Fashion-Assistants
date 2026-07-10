# ============================================================
# gemini_chatbot.py — Conversational Fashion Assistant
# ============================================================
# Uses Google Gemini API for two tasks:
#   A) Intent extraction  — parse user message → structured query
#   B) Explanation        — generate styled rationale for recommendations
# ============================================================

import json
import re
from typing import Dict, List, Optional, Any
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.config import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_TEMPERATURE, GEMINI_MAX_TOKENS

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# ── System prompts ───────────────────────────────────────────

INTENT_SYSTEM_PROMPT = """You are a fashion assistant specializing in outfit analysis.

Your task is to extract structured information from a user's fashion request.

Given the user message, return a JSON object with these fields:
{
  "gender": "men" | "women" | null,
  "age": <integer or null>,
  "occasion": <string or null>,
  "style": <string or null>,
  "items_mentioned": [<list of clothing items mentioned>],
  "max_price_inr": <integer or null, e.g. if 'under 1000' then 1000>,
  "query_intent": "find_item" | "complete_outfit" | "general",
  "seed_item": <specific item name if user mentions one specific item, else null>,
  "enriched_query": <a concise 1-2 sentence description of what to search for>
}

CRITICAL RULES:
1. If the user asks for specific items (e.g. "shoes under 1000", "show shirts"), set `query_intent` to "find_item".
2. If the user asks for a full outfit (e.g. "wedding outfit", "casual look"), set `query_intent` to "complete_outfit".
3. Extract `max_price_inr` purely as an integer (e.g. 1000).

Respond with ONLY the JSON object, no extra text.
"""

EXPLANATION_SYSTEM_PROMPT = """You are a professional fashion stylist AI with expertise in outfit coordination.

When given a list of outfit items, generate a natural, engaging explanation of why these pieces work well together.

Guidelines:
- Be specific: mention actual colors, fits, and fabrics from the item descriptions
- Reference fashion principles (contrast, color theory, occasion appropriateness)
- Keep it 2-3 sentences, conversational yet knowledgeable
- Be encouraging and positive
- Mention the occasion suitability
- Do NOT use generic filler phrases like "great choice" or "perfect outfit"

Respond with just the explanation paragraph.
"""

CHAT_SYSTEM_PROMPT = """You are an AI-powered fashion assistant for an Indian fashion recommendation platform.

You help users find complete outfit recommendations based on their preferences, occasions, and style.

Your personality:
- Friendly, knowledgeable, and stylish
- Give specific, actionable suggestions
- Ask clarifying questions when needed (gender, occasion, budget)
- Reference the actual recommended items in your responses
- Use fashion terminology naturally

When you present recommendations, describe them naturally as if you're a human stylist.
Keep responses concise (under 150 words) unless the user asks for details.

Remember: you have access to a database of 68 fashion products from Myntra, Ajio, and Nykaa with images.
"""


class GeminiFashionChatbot:
    """
    Conversational fashion assistant powered by Gemini.

    Maintains conversation history for multi-turn dialogue and provides:
    - Intent parsing (structured query extraction)
    - Outfit explanation generation
    - Full conversational responses with embedded recommendations
    """

    def __init__(self):
        self.chat_history: List[Dict[str, str]] = []
        self._client  = None
        self._chat    = None
        self._ready   = False

    # ── Initialisation ───────────────────────────────────────

    def initialize(self) -> bool:
        """Set up the Gemini client. Returns True if successful."""
        if not GEMINI_AVAILABLE:
            print("[Gemini] google-generativeai not installed. Using fallback mode.")
            return False
        if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
            print("[Gemini] API key not configured. Using fallback mode.")
            return False
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            self._client = genai.GenerativeModel(
                model_name=GEMINI_MODEL,
                system_instruction=CHAT_SYSTEM_PROMPT,
                generation_config=genai.types.GenerationConfig(
                    temperature=GEMINI_TEMPERATURE,
                    max_output_tokens=GEMINI_MAX_TOKENS,
                ),
            )
            self._chat  = self._client.start_chat(history=[])
            self._ready = True
            print(f"[Gemini] Initialized with model '{GEMINI_MODEL}' OK")
            return True
        except Exception as e:
            print(f"[Gemini] Initialization failed: {e}")
            return False

    # ── A) Intent Extraction ─────────────────────────────────

    def extract_intent(self, user_message: str) -> Dict[str, Any]:
        """
        Parse a natural-language message into a structured intent dict.
        Falls back to rule-based extraction if Gemini is unavailable.

        Returns:
            {
              gender, age, occasion, style, items_mentioned,
              max_price_inr, query_intent, seed_item, enriched_query
            }
        """
        if self._ready:
            return self._gemini_extract_intent(user_message)
        return self._rule_based_intent(user_message)

    def _gemini_extract_intent(self, message: str) -> Dict[str, Any]:
        """Use Gemini to extract structured intent."""
        try:
            extractor = genai.GenerativeModel(
                model_name=GEMINI_MODEL,
                system_instruction=INTENT_SYSTEM_PROMPT,
            )
            response = extractor.generate_content(message)
            raw = response.text.strip()
            # Strip markdown code fences if present
            raw = re.sub(r"```(?:json)?\n?", "", raw).strip().rstrip("```").strip()
            intent = json.loads(raw)
            # Ensure all keys exist
            defaults = {
                "gender": None, "age": None, "occasion": None, "style": None,
                "items_mentioned": [], "max_price_inr": None,
                "query_intent": "general", "seed_item": None,
                "enriched_query": message,
            }
            return {**defaults, **intent}
        except Exception as e:
            # Check for Quota Exceeded (429)
            if "429" in str(e) or "Quota" in str(e):
                print(f"[Gemini] Quota exceeded during intent extraction. Using local rule-based fallback.")
            else:
                print(f"[Gemini] Intent extraction error: {e}")
            return self._rule_based_intent(message)

    def _rule_based_intent(self, message: str) -> Dict[str, Any]:
        """Fallback rule-based intent extraction when Gemini is unavailable."""
        msg = message.lower()

        # Gender detection
        gender = None
        if any(w in msg for w in ["women", "woman", "female", "girl", "her", "ladies"]):
            gender = "women"
        elif any(w in msg for w in ["men", "man", "male", "boy", "his", "gentleman"]):
            gender = "men"

        # Occasion detection
        occasion = None
        occasion_map = {
            "office": ["office", "work", "business meeting", "corporate", "professional"],
            "interview": ["interview"],
            "wedding": ["wedding", "shaadi", "reception"],
            "party": ["party", "night out", "cocktail", "evening"],
            "casual": ["casual", "everyday", "daily", "weekend"],
            "beach": ["beach", "vacation", "holiday", "travel"],
            "date": ["date", "dinner date", "romantic"],
            "outdoor": ["outdoor", "sports", "gym", "workout"],
        }
        for occ, keywords in occasion_map.items():
            if any(k in msg for k in keywords):
                occasion = occ
                break

        # Style detection
        style = None
        if any(w in msg for w in ["formal", "professional", "corporate"]):
            style = "formal"
        elif any(w in msg for w in ["casual", "relaxed", "everyday"]):
            style = "casual"
        elif any(w in msg for w in ["smart casual", "smart-casual"]):
            style = "smart-casual"
        elif any(w in msg for w in ["streetwear", "street style", "urban"]):
            style = "streetwear"

        # Age detection
        age = None
        age_match = re.search(r"\b(\d{1,2})[- ]?(?:year|yr|y\.?o\.?)\b", msg)
        if age_match:
            age = int(age_match.group(1))

        # Item mentions
        item_keywords = [
            "shirt", "t-shirt", "jeans", "trousers", "blazer", "suit",
            "dress", "skirt", "top", "kurta", "saree", "sneakers",
            "heels", "loafers", "jacket", "sweatshirt", "shorts",
            "shoes", "footwear", "watch", "accessories"
        ]
        items_mentioned = [w for w in item_keywords if w in msg]

        # Price extraction
        max_price_inr = None
        price_match = re.search(r"(?:under|below|max|for)\s*(?:rs|inr|₹)?\s*(\d+)", msg)
        if price_match:
            max_price_inr = int(price_match.group(1))

        # Query intent
        if len(items_mentioned) > 0:
            query_intent = "find_item"
            seed_item = items_mentioned[0]
        elif "outfit" in msg or "look" in msg:
            query_intent = "complete_outfit"
            seed_item = None
        else:
            query_intent = "complete_outfit" # Default to outfit if ambiguous
            seed_item = None

        enriched_query = message
        if gender and occasion:
            enriched_query += f" {gender} {occasion} outfit"

        return {
            "gender": gender,
            "age": age,
            "occasion": occasion,
            "style": style,
            "items_mentioned": items_mentioned,
            "max_price_inr": max_price_inr,
            "query_intent": query_intent,
            "seed_item": seed_item,
            "enriched_query": enriched_query,
        }

    # ── B) Explanation Generation ────────────────────────────

    def generate_outfit_explanation(
        self,
        outfit_items: List[Dict],
        occasion:     Optional[str] = None,
        user_context: Optional[str] = None,
    ) -> str:
        """
        Generate a natural-language styling rationale for a recommended outfit.

        Args:
            outfit_items: List of product metadata dicts.
            occasion:     Optional occasion string.
            user_context: Optional user's original query for personalization.

        Returns:
            A 2-3 sentence styling explanation.
        """
        if self._ready:
            return self._gemini_explain(outfit_items, occasion, user_context)
        return self._rule_based_explanation(outfit_items, occasion)

    def _gemini_explain(
        self,
        items: List[Dict],
        occasion: Optional[str],
        user_context: Optional[str],
    ) -> str:
        """Use Gemini to generate a styling explanation."""
        try:
            item_descriptions = "\n".join([
                f"- {i.get('name', '')} ({i.get('category_label', '')})"
                f" — {i.get('description', '')[:100]}"
                for i in items
            ])
            prompt = f"Outfit items:\n{item_descriptions}"
            if occasion:
                prompt += f"\nOccasion: {occasion}"
            if user_context:
                prompt += f"\nUser requested: {user_context}"

            explainer = genai.GenerativeModel(
                model_name=GEMINI_MODEL,
                system_instruction=EXPLANATION_SYSTEM_PROMPT,
            )
            response = explainer.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"[Gemini] Explanation error: {e}")
            return self._rule_based_explanation(items, occasion)

    def _rule_based_explanation(self, items: List[Dict], occasion: Optional[str]) -> str:
        """Fallback explanation when Gemini is unavailable."""
        if not items:
            return "Here are your recommended items."
            
        primary = items[0]
        gender_str = primary.get("gender", "")
        if gender_str:
            gender_str = f"{gender_str}'s "
            
        name = primary.get("name", "product")
        cat = primary.get("category_label", "item").lower()
        price = sum(float(i.get("price_inr", 0)) for i in items)
        
        occ_str = occasion if occasion else "everyday casual"
        
        if len(items) == 1:
            return f"This {gender_str}{name} matches your {cat} search and falls within your ₹{price:.0f} budget. Its versatile design makes it suitable for {occ_str} wear."
        else:
            return f"This {gender_str}outfit matches your search and falls within your ₹{price:.0f} budget. Its versatile design makes it suitable for {occ_str} wear."

    # ── C) Full Conversational Chat ──────────────────────────

    def chat(
        self,
        user_message:   str,
        outfit_context: Optional[List[Dict]] = None,
    ) -> str:
        """
        Send a message to the conversational chat and get a response.
        Optionally pass recommended outfit items to include in context.

        Returns the assistant's response string.
        """
        # Build enriched message with outfit context
        enriched = user_message
        if outfit_context:
            items_str = ", ".join([
                f"{i.get('name','')} (₹{i.get('price_inr',0):.0f})"
                for i in outfit_context
            ])
            enriched += f"\n\n[Recommended outfit: {items_str}]"

        self.chat_history.append({"role": "user", "content": user_message})

        if self._ready:
            try:
                response = self._chat.send_message(enriched)
                reply = response.text.strip()
            except Exception as e:
                # Catch quota errors and other exceptions gracefully
                reply = self._fallback_response(outfit_context)
        else:
            reply = self._fallback_response(outfit_context)

        self.chat_history.append({"role": "assistant", "content": reply})
        return reply

    def _fallback_response(self, context_items: Optional[List[Dict]]) -> str:
        """A user-friendly fallback response when Gemini is unavailable."""
        return "✨ Curated recommendations selected from our fashion catalog and tailored to your preferences."

    def reset_conversation(self):
        """Clear chat history and start fresh."""
        self.chat_history = []
        if self._ready:
            self._chat = self._client.start_chat(history=[])

    def get_history(self) -> List[Dict[str, str]]:
        return self.chat_history.copy()

    def is_ready(self) -> bool:
        return self._ready


# ── Standalone test ──────────────────────────────────────────
if __name__ == "__main__":
    bot = GeminiFashionChatbot()
    ready = bot.initialize()
    print(f"Gemini ready: {ready}")

    # Test intent extraction
    test_messages = [
        "I need an outfit for a business meeting",
        "Suggest a smart casual outfit for a dinner date",
        "I am a 22-year-old male looking for a casual summer outfit",
        "I need something stylish for a beach vacation",
    ]
    print("\n=== Intent Extraction Tests ===")
    for msg in test_messages:
        intent = bot.extract_intent(msg)
        print(f"\nInput: '{msg}'")
        print(f"  → gender={intent['gender']}, occasion={intent['occasion']}, "
              f"style={intent['style']}, intent={intent['query_intent']}")
        print(f"  → enriched: {intent['enriched_query']}")

    # Test explanation
    test_items = [
        {"name": "White Formal Shirt", "category_label": "Formal Shirts",
         "description": "White slim fit formal shirt with cutaway collar"},
        {"name": "Navy Blue Trousers", "category_label": "Trousers",
         "description": "Navy blue slim fit formal trousers"},
        {"name": "Brown Leather Loafers", "category_label": "Loafers",
         "description": "Classic brown leather slip-on loafers"},
    ]
    print("\n=== Explanation Test ===")
    explanation = bot.generate_outfit_explanation(test_items, occasion="office")
    print(f"Explanation: {explanation}")

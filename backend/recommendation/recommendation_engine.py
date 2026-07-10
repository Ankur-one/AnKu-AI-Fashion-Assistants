# ============================================================
# recommendation_engine.py — Main outfit recommendation pipeline
# ============================================================
# Orchestrates FAISS search + compatibility scoring to produce
# complete, ranked outfit suggestions.
# ============================================================

import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.config import TOP_K_CANDIDATES, TOP_K_OUTFITS
from backend.embeddings.faiss_index import FashionFAISSIndex
from backend.recommendation.compatibility_engine import OutfitCompatibilityEngine


# ── Data classes (plain dicts for simplicity) ────────────────
#
# UserProfile  = { gender, age, occasion, style, max_price }
# Outfit       = { items: [product_meta], score, explanation }


class RecommendationEngine:
    """
    Main entry point for outfit recommendations.

    Two modes
    ---------
    1. recommend_from_item(product_id)
       Given one seed item, complete the outfit by finding compatible
       pieces for the remaining slots.

    2. recommend_from_profile(user_profile, query_text)
       Given a user description / chat query, generate complete outfit
       suggestions from scratch.
    """

    def __init__(self):
        self.faiss       = FashionFAISSIndex()
        self.compat      = OutfitCompatibilityEngine()
        self._initialized = False

    # ── Initialisation ───────────────────────────────────────

    def initialize(self, force_rebuild: bool = False):
        """Load/build indexes. Call once before any recommendations."""
        if self._initialized:
            return
        print("[RecEngine] Initialising ...")
        self.faiss.build_index(force_rebuild=force_rebuild)
        self.compat.load()
        self._initialized = True
        print("[RecEngine] Ready OK")

    def ensure_ready(self):
        if not self._initialized:
            self.initialize()

    # ── Mode 1: Complete an outfit from a seed item ──────────

    def recommend_from_item(
        self,
        product_id: str,
        gender:     Optional[str] = None,
        occasion:   Optional[str] = None,
        max_price:  Optional[float] = None,
        top_k:      int = TOP_K_OUTFITS,
    ) -> List[Dict]:
        """
        Given a product_id, recommend complete outfits containing that item.

        Returns a list of up to `top_k` outfit dicts:
          { 'items': [...], 'score': float, 'explanation': str }
        """
        self.ensure_ready()

        # Fetch seed item
        seed_meta = self._get_meta_by_id(product_id)
        if not seed_meta:
            return []

        seed_embed = self.faiss.get_product_embedding(product_id)
        seed_slot  = seed_meta.get("slot", "other")

        # Infer filters from seed if not explicitly provided
        if not gender:
            gender = seed_meta.get("gender", None)
        if not occasion:
            occasion = seed_meta.get("occasion", None)

        # Determine which slots still need filling
        remaining_slots = self._slots_to_fill(seed_slot)

        # For each remaining slot, get candidates via FAISS
        slot_candidates = {}
        for slot in remaining_slots:
            candidates = self.faiss.search_slot(
                query_embedding=seed_embed,
                slot=slot,
                gender=gender,
                occasion=occasion,
                max_price=max_price,
                top_k=TOP_K_CANDIDATES,
            )
            slot_candidates[slot] = candidates

        # Assemble outfits from cross-product of top candidates
        raw_outfits = self._assemble_outfits(seed_meta, slot_candidates)

        # Score and rank
        return self._rank_outfits(raw_outfits, top_k)

    # ── Mode 2: Generate outfit from user profile / query ────

    def recommend_from_profile(
        self,
        query_text:  str,
        gender:      Optional[str] = None,
        age:         Optional[int] = None,
        occasion:    Optional[str] = None,
        style:       Optional[str] = None,
        max_price:   Optional[float] = None,
        top_k:       int = TOP_K_OUTFITS,
    ) -> List[Dict]:
        """
        Generate full outfit recommendations from a natural language query
        and optional user profile information.

        Returns list of outfit dicts sorted by compatibility score.
        """
        self.ensure_ready()

        # Enrich query with profile context
        enriched_query = self._build_query(query_text, gender, age, occasion, style)

        # Get query embedding
        self.faiss.embedder.load_model()
        query_embed = self.faiss.embedder.encode_query(enriched_query)

        # Normalize occasion to match dataset
        if occasion:
            occ_lower = occasion.lower()
            if occ_lower in ("date", "date night", "romantic"):
                occasion = "party"
            elif occ_lower not in ("office", "wedding", "casual", "sports", "vacation", "party", "festive", "winter"):
                occasion = "casual"  # Fallback to casual if unknown

        # Build filters
        filters = {}
        if gender:
            filters["gender"] = gender
        if occasion:
            filters["occasion"] = occasion
        if max_price:
            filters["max_price"] = max_price

        # For each slot, retrieve top candidates
        all_slots = ["topwear", "bottomwear", "footwear"]
        slot_candidates = {}
        for slot in all_slots:
            slot_filters = dict(filters)
            slot_filters["slot"] = slot
            candidates = self.faiss.search(
                query_embed,
                top_k=TOP_K_CANDIDATES,
                filters=slot_filters,
            )
            slot_candidates[slot] = candidates

        # Also try accessories
        acc_filters = dict(filters)
        acc_filters["slot"] = "accessory"
        acc_candidates = self.faiss.search(query_embed, top_k=5, filters=acc_filters)
        slot_candidates["accessory"] = acc_candidates
        
        # --- DEBUG LOGGING ---
        print("\n=== DEBUG: recommend_from_profile ===")
        print(f"Normalized Occasion: {occasion}")
        print(f"Style: {style}")
        print(f"Gender: {gender}")
        print(f"Max Price: {max_price}")
        print(f"Candidate Topwear: {len(slot_candidates.get('topwear', []))}")
        print(f"Candidate Bottomwear: {len(slot_candidates.get('bottomwear', []))}")
        print(f"Candidate Footwear: {len(slot_candidates.get('footwear', []))}")
        print(f"Candidate Accessory: {len(slot_candidates.get('accessory', []))}")
        print("======================================\n")

        # Assemble outfits
        raw_outfits = self._assemble_from_slots(slot_candidates, max_price=max_price)

        return self._rank_outfits(raw_outfits, top_k)

    # ── Search by text (for simple queries) ─────────────────

    def search_products(
        self,
        query_text: str,
        product_type: Optional[str] = None,
        gender: Optional[str] = None,
        max_price: Optional[float] = None,
        top_k: int = 10,
    ) -> List[Dict]:
        """Dedicated product search pipeline returning only matching products."""
        self.ensure_ready()
        
        filters = {}
        if gender:
            filters["gender"] = gender
        if max_price is not None:
            filters["max_price"] = max_price
            
        # Map product_type to slot if provided
        if product_type:
            from backend.config import TOPWEAR_CATEGORIES, BOTTOMWEAR_CATEGORIES, FOOTWEAR_CATEGORIES, ACCESSORY_CATEGORIES
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
            # We don't strictly set a filter if the mapping fails, to allow FAISS to fallback to text

        return self.faiss.search_by_text(query_text, top_k=top_k, filters=filters)

    # ── Find similar items ───────────────────────────────────

    def find_similar(self, product_id: str, top_k: int = 5) -> List[Dict]:
        """Find products visually/semantically similar to a given item."""
        self.ensure_ready()
        return self.faiss.search_by_product_id(product_id, top_k=top_k)

    # ── Private helpers ──────────────────────────────────────

    def _get_meta_by_id(self, product_id: str) -> Optional[Dict]:
        all_meta = self.faiss.get_all_meta()
        for m in all_meta:
            if m["id"] == product_id:
                return m
        return None

    def _slots_to_fill(self, seed_slot: str) -> List[str]:
        """Return the slots that need candidates given the seed's slot."""
        all_slots = ["topwear", "bottomwear", "footwear"]
        return [s for s in all_slots if s != seed_slot]

    def _build_query(
        self,
        query_text: str,
        gender:     Optional[str],
        age:        Optional[int],
        occasion:   Optional[str],
        style:      Optional[str],
    ) -> str:
        """Compose an enriched text query for CLIP encoding."""
        parts = [query_text]
        if gender:
            parts.append(f"{gender}'s outfit")
        if occasion:
            parts.append(f"for {occasion}")
        if style:
            parts.append(f"{style} style")
        if age:
            if age < 25:
                parts.append("trendy youthful")
            elif age < 40:
                parts.append("smart professional")
            else:
                parts.append("classic sophisticated")
        return " ".join(parts)

    def _assemble_outfits(
        self,
        seed_meta:        Dict,
        slot_candidates:  Dict[str, List[Dict]],
        max_combinations: int = 20,
    ) -> List[Dict[str, Any]]:
        """Assemble outfit combinations with the seed item fixed."""
        outfits = []
        slots   = sorted(slot_candidates.keys())

        if not slots:
            return [{"items": [seed_meta]}]

        # Use top-N per slot to limit combinations
        N = 4
        capped = {s: slot_candidates[s][:N] for s in slots}

        # Generate cross-product combinations
        from itertools import product as iproduct
        slot_lists = [capped[s] for s in slots]
        for combo in iproduct(*slot_lists):
            if len(outfits) >= max_combinations:
                break
            items = [seed_meta] + list(combo)
            # De-dupe by product id
            seen = set()
            deduped = []
            for it in items:
                if it["id"] not in seen:
                    seen.add(it["id"])
                    deduped.append(it)
            outfits.append({"items": deduped})

        return outfits

    def _assemble_from_slots(
        self,
        slot_candidates: Dict[str, List[Dict]],
        max_combinations: int = 30,
        max_price: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Assemble outfit combinations from slot candidates, enforcing total budget."""
        from itertools import product as iproduct

        main_slots = ["topwear", "bottomwear", "footwear"]
        acc_candidates = slot_candidates.get("accessory", [])

        N = 3  # cap per slot
        main_lists = []
        for s in main_slots:
            candidates = slot_candidates.get(s, [])[:N]
            if candidates:
                main_lists.append(candidates)
            else:
                # If a slot has NO candidates (e.g. filtered out by budget),
                # we append a list with `None` so itertools.product still works
                # and we can at least return a partial outfit.
                main_lists.append([None])

        outfits = []
        for combo in iproduct(*main_lists):
            if len(outfits) >= max_combinations:
                break
            # Filter out the None values we inserted for empty slots
            items = [it for it in combo if it is not None]
            
            if not items:
                continue

            # Optionally add best accessory
            if acc_candidates:
                items.append(acc_candidates[0])
            # De-dupe
            seen = set()
            deduped = []
            for it in items:
                if it["id"] not in seen:
                    seen.add(it["id"])
                    deduped.append(it)
            
            # Enforce total budget
            if max_price is not None:
                total_price = sum(float(i.get("price_inr", 0)) for i in deduped)
                if total_price > max_price:
                    # If it exceeds the budget, skip this combination
                    # Or we could try removing the accessory to see if it fits
                    if acc_candidates and len(deduped) > 1:
                        total_without_acc = total_price - float(deduped[-1].get("price_inr", 0))
                        if total_without_acc <= max_price:
                            deduped.pop()
                        else:
                            continue
                    else:
                        continue
            
            outfits.append({"items": deduped})

        return outfits

    def _rank_outfits(
        self,
        raw_outfits: List[Dict],
        top_k:       int,
    ) -> List[Dict]:
        """Score all candidate outfits iteratively to promote diversity and return top-k."""
        # Preload embeddings for the scoring
        all_ids    = set()
        for o in raw_outfits:
            for item in o["items"]:
                all_ids.add(item["id"])

        embed_map: Dict[str, np.ndarray] = {}
        for pid in all_ids:
            e = self.faiss.get_product_embedding(pid)
            if e is not None:
                embed_map[pid] = e

        final_scored = []
        remaining = list(raw_outfits)
        
        while remaining and len(final_scored) < top_k:
            best_idx = -1
            best_score = -1.0
            best_expl = ""
            
            previous_outfits = [s["items"] for s in final_scored]
            
            for i, outfit in enumerate(remaining):
                score = self.compat.score_outfit(outfit["items"], embeddings=embed_map, previous_outfits=previous_outfits)
                
                # Filter out bad matches (e.g. mixed genders will score very low)
                if score < 0.5:
                    continue
                    
                if score > best_score:
                    best_score = score
                    best_idx = i
                    
            if best_idx != -1:
                best_outfit = remaining.pop(best_idx)
                expl = self.compat.get_compatibility_explanation(best_outfit["items"])
                final_scored.append({
                    "items":       best_outfit["items"],
                    "score":       best_score,
                    "explanation": expl,
                })
            else:
                break
                
        return final_scored


# ── Standalone test ──────────────────────────────────────────
if __name__ == "__main__":
    engine = RecommendationEngine()
    engine.initialize()

    print("\n=== Mode 1: Complete outfit from seed item ===")
    # Get first product id
    all_meta = engine.faiss.get_all_meta()
    seed_id  = all_meta[0]["id"]
    seed_name = all_meta[0]["name"]
    print(f"Seed: {seed_name} ({seed_id})")

    outfits = engine.recommend_from_item(seed_id, top_k=2)
    for i, o in enumerate(outfits, 1):
        print(f"\nOutfit {i} (score={o['score']:.3f}):")
        for item in o["items"]:
            print(f"  • {item['name']} [{item.get('slot','')}] ₹{item.get('price_inr',0)}")
        print(f"  → {o['explanation']}")

    print("\n=== Mode 2: From user profile ===")
    outfits2 = engine.recommend_from_profile(
        query_text="business meeting formal wear",
        gender="men",
        occasion="office",
        top_k=2,
    )
    for i, o in enumerate(outfits2, 1):
        print(f"\nOutfit {i} (score={o['score']:.3f}):")
        for item in o["items"]:
            print(f"  • {item['name']} [{item.get('slot','')}] ₹{item.get('price_inr',0)}")
        print(f"  → {o['explanation']}")

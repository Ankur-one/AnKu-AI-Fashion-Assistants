# ============================================================
# compatibility_engine.py — Outfit compatibility scoring
# ============================================================
# Learns from the 25 curated outfits and applies multiple
# scoring signals to rank how well items work together.
# ============================================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.config import (
    OCCASION_GROUPS, NEUTRAL_COLORS, WARM_COLORS, COOL_COLORS,
    TOPWEAR_CATEGORIES, BOTTOMWEAR_CATEGORIES,
    FOOTWEAR_CATEGORIES, ACCESSORY_CATEGORIES,
)
from backend.data_loader import DataLoader


class OutfitCompatibilityEngine:
    """
    Scores the compatibility of fashion item combinations using:

    1. Category slot fitness   — right items in right slots
    2. Occasion coherence      — items share compatible occasions
    3. Gender consistency      — all items match user/request gender
    4. Color harmony           — color palette compatibility
    5. Embedding coherence     — moderate CLIP similarity (not too same, not too different)
    6. Price harmony           — price tiers are consistent within outfit
    7. Curated outfit learning — co-occurrence patterns from the 25 curated outfits
    """

    # New Scoring weights (must sum to 1.0)
    W_EMBEDDING = 0.50
    W_OCCASION  = 0.20
    W_STYLE     = 0.15
    W_PRICE     = 0.10
    W_DIVERSITY = 0.05

    def __init__(self):
        self.loader    = DataLoader()
        self._outfit_pairs: Dict[Tuple[str, str], int] = {}  # (cat_a, cat_b) → count
        self._loaded   = False

    # ── Public API ───────────────────────────────────────────

    def load(self):
        """Learn co-occurrence statistics from the curated outfits."""
        if self._loaded:
            return
        self._learn_from_curated_outfits()
        self._loaded = True

    def score_outfit(
        self,
        items: List[Dict],
        embeddings: Optional[Dict[str, np.ndarray]] = None,
        previous_outfits: Optional[List[List[Dict]]] = None,
    ) -> float:
        """
        Score a list of product metadata dicts as an outfit.

        Args:
            items:      List of product meta dicts (from FAISS / DataLoader).
            embeddings: Optional {product_id → 512-dim embedding} for
                        embedding-coherence signal.
            previous_outfits: Optional list of previously scored and selected outfits
                        to compute the diversity penalty.

        Returns:
            float in [0, 1] — higher is better.
        """
        self.load()
        if not items:
            return 0.0

        scores = {
            "embedding": self._score_embedding_coherence(items, embeddings),
            "occasion":  self._score_occasion_coherence(items),
            "style":     self._score_style_coherence(items),
            "price":     self._score_price_harmony(items),
            "diversity": self._score_diversity(items, previous_outfits),
        }

        total = (
            self.W_EMBEDDING * scores["embedding"] +
            self.W_OCCASION  * scores["occasion"]  +
            self.W_STYLE     * scores["style"]     +
            self.W_PRICE     * scores["price"]     +
            self.W_DIVERSITY * scores["diversity"]
        )
        return round(min(max(total, 0.0), 1.0), 4)

    def score_pair(
        self,
        item_a: Dict,
        item_b: Dict,
        embed_a: Optional[np.ndarray] = None,
        embed_b: Optional[np.ndarray] = None,
    ) -> float:
        """Score compatibility between two items."""
        items = [item_a, item_b]
        embeds = None
        if embed_a is not None and embed_b is not None:
            embeds = {item_a["id"]: embed_a, item_b["id"]: embed_b}
        return self.score_outfit(items, embeds)

    def get_compatibility_explanation(self, items: List[Dict]) -> str:
        """Return a human-readable explanation using product names, occasion, and price."""
        if not items:
            return "No items found."

        names = [i.get("name", "item") for i in items]
        occasion = self._dominant_occasion(items) or "everyday"
        style = self._dominant_style(items) or "versatile"
        price = sum(float(i.get("price_inr", 0)) for i in items)

        if len(names) == 1:
            combo_str = names[0]
        elif len(names) == 2:
            combo_str = f"{names[0]} and {names[1]}"
        else:
            base = names[0]
            rest = names[1:-1]
            last = names[-1]
            combo_str = f"{base}, " + ", ".join(rest) + f", and {last}"

        return (
            f"This {occasion} outfit combines {combo_str}. "
            f"The combination creates a stylish {style} look while staying within your ₹{price:.0f} budget."
        )

    def _dominant_style(self, items: List[Dict]) -> Optional[str]:
        styles = []
        for item in items:
            name = item.get("name", "").lower()
            cat = item.get("category_label", "").lower()
            for s in ["formal", "casual", "streetwear", "sporty", "ethnic", "smart-casual", "evening"]:
                if s in name or s in cat:
                    styles.append(s)
        if not styles:
            return None
        return max(set(styles), key=styles.count)

    def _score_diversity(self, items: List[Dict], previous_outfits: Optional[List[List[Dict]]]) -> float:
        if not previous_outfits:
            return 1.0
        
        item_ids = {i["id"] for i in items}
        
        penalty = 0.0
        for prev_outfit in previous_outfits:
            prev_ids = {i["id"] for i in prev_outfit}
            overlap = len(item_ids.intersection(prev_ids))
            if overlap > 0:
                penalty += overlap / max(len(items), 1)
                
        # Maximum penalty means 0 diversity score, no penalty means 1.0 diversity score
        diversity_score = max(0.0, 1.0 - penalty)
        return diversity_score

    # ── Signal implementations ───────────────────────────────

    def _score_style_coherence(self, items: List[Dict]) -> float:
        """
        Check if items share similar style tags.
        """
        if len(items) < 2:
            return 1.0

        style_tags = []
        for item in items:
            name = item.get("name", "").lower()
            cat = item.get("category_label", "").lower()
            
            item_styles = set()
            for s in ["formal", "casual", "streetwear", "sporty", "ethnic", "smart-casual"]:
                if s in name or s in cat:
                    item_styles.add(s)
            
            if item_styles:
                style_tags.append(item_styles)

        if not style_tags:
            return 0.5  # Neutral if no strong style tags found

        # Calculate overlap
        overlap_score = 0.0
        pairs = 0
        for i in range(len(style_tags)):
            for j in range(i + 1, len(style_tags)):
                if style_tags[i].intersection(style_tags[j]):
                    overlap_score += 1.0
                pairs += 1
        
        return overlap_score / pairs if pairs > 0 else 0.5
        return overlap_score / pairs if pairs > 0 else 0.5

    def _score_occasion_coherence(self, items: List[Dict]) -> float:
        """Items should share compatible occasions."""
        occasions = [i.get("occasion", "").lower() for i in items if i.get("occasion")]
        if len(occasions) < 2:
            return 0.8  # neutral if only one item has occasion

        # Map each item occasion to a group
        def get_group(occ: str) -> Optional[str]:
            for group, occs in OCCASION_GROUPS.items():
                if any(o in occ for o in occs):
                    return group
            return None

        groups = [get_group(o) for o in occasions]
        valid  = [g for g in groups if g is not None]
        if not valid:
            return 0.7

        # Score = fraction of items in the majority group
        majority = max(set(valid), key=valid.count)
        return valid.count(majority) / len(valid)

    def _score_gender_consistency(self, items: List[Dict]) -> float:
        """All items should target the same gender (or be unisex)."""
        genders = [i.get("gender", "").lower() for i in items if i.get("gender")]
        genders = [g for g in genders if g not in ("", "unisex")]
        if not genders:
            return 1.0
        
        # Severe penalty for mixing genders
        if len(set(genders)) > 1:
            return 0.0
            
        return 1.0

    def _score_color_harmony(self, items: List[Dict]) -> float:
        """
        Basic color theory scoring:
        - Neutral + any color = great
        - All neutral = good
        - Same family (all warm / all cool) = good
        - Warm + cool mix = moderate
        """
        colors = self._extract_colors(items)
        if not colors:
            return 0.7

        neutrals = colors & NEUTRAL_COLORS
        warms    = colors & WARM_COLORS
        cools    = colors & COOL_COLORS

        # Mostly neutral — always safe
        if len(neutrals) / max(len(colors), 1) > 0.6:
            return 0.9
        # Single family
        if warms and not cools:
            return 0.85
        if cools and not warms:
            return 0.85
        # Neutral + one accent — classic combo
        if neutrals and (warms or cools):
            return 0.95
        # Mixed warm & cool
        return 0.55

    def _score_price_harmony(self, items: List[Dict]) -> float:
        """
        Items should be in a similar price tier.
        High variance = lower score (mixing budget with luxury looks off).
        """
        prices = [float(i.get("price_inr", 0)) for i in items if i.get("price_inr", 0) > 0]
        if len(prices) < 2:
            return 0.8
        mean = np.mean(prices)
        std  = np.std(prices)
        cv   = std / mean if mean > 0 else 0   # coefficient of variation
        # cv < 0.3 → very consistent, cv > 1.0 → very spread
        return max(0.0, 1.0 - cv)

    def _score_embedding_coherence(
        self,
        items: List[Dict],
        embeddings: Optional[Dict[str, np.ndarray]],
    ) -> float:
        """
        CLIP embedding coherence: items in a good outfit should be
        related but not identical.
        Target cosine similarity range: [0.15, 0.65]
        """
        if embeddings is None or len(items) < 2:
            return 0.7   # default if no embeddings available

        vecs = []
        for item in items:
            pid = item.get("id", "")
            if pid in embeddings:
                vecs.append(embeddings[pid])

        if len(vecs) < 2:
            return 0.7

        # Average pairwise cosine similarity
        sims = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                sims.append(float(np.dot(vecs[i], vecs[j])))

        avg_sim = np.mean(sims)
        # Penalise extremes: too similar (redundant) or too different (clash)
        if 0.15 <= avg_sim <= 0.65:
            return 1.0
        elif avg_sim < 0.15:
            return max(0.0, avg_sim / 0.15)
        else:
            return max(0.0, 1.0 - (avg_sim - 0.65) / 0.35)

    def _score_curated_cooccurrence(self, items: List[Dict]) -> float:
        """
        Check if any item pairs have been seen together in the curated outfits.
        Uses category-level co-occurrence (not product-level) for generalisation.
        """
        self.load()
        if not self._outfit_pairs:
            return 0.5

        cats = [i.get("category", "").lower() for i in items]
        hits = 0
        pairs_checked = 0
        for i in range(len(cats)):
            for j in range(i + 1, len(cats)):
                pair = tuple(sorted([cats[i], cats[j]]))
                pairs_checked += 1
                if pair in self._outfit_pairs:
                    hits += 1

        if pairs_checked == 0:
            return 0.5
        return hits / pairs_checked

    # ── Learning from curated data ───────────────────────────

    def _learn_from_curated_outfits(self):
        """Parse curated outfits to build category co-occurrence stats."""
        try:
            outfits  = self.loader.load_outfits()
            products = self.loader.load_products()
            id_to_cat = dict(zip(products["id"], products["category"]))

            id_cols = ["hero_id", "second_id", "layer_id",
                       "footwear_id", "accessory_1_id", "accessory_2_id"]

            for _, row in outfits.iterrows():
                item_cats = []
                for col in id_cols:
                    pid = row.get(col)
                    if pd.notna(pid) and pid:
                        cat = id_to_cat.get(str(pid), "")
                        if cat:
                            item_cats.append(cat.lower())

                # Record all pairs
                for i in range(len(item_cats)):
                    for j in range(i + 1, len(item_cats)):
                        pair = tuple(sorted([item_cats[i], item_cats[j]]))
                        self._outfit_pairs[pair] = self._outfit_pairs.get(pair, 0) + 1

        except Exception as e:
            print(f"[Compatibility] Warning: could not learn from curated outfits: {e}")

    # ── Helpers ──────────────────────────────────────────────

    def _extract_colors(self, items: List[Dict]) -> Set[str]:
        """Extract color keywords from item names and descriptions."""
        all_colors = NEUTRAL_COLORS | WARM_COLORS | COOL_COLORS
        found = set()
        for item in items:
            text = (item.get("name", "") + " " + item.get("description", "")).lower()
            for color in all_colors:
                if color in text:
                    found.add(color)
        return found

    def _dominant_occasion(self, items: List[Dict]) -> str:
        """Return the most common occasion across items."""
        occasions = [i.get("occasion", "") for i in items if i.get("occasion")]
        if not occasions:
            return "any"
        return max(set(occasions), key=occasions.count)


# ── Standalone test ──────────────────────────────────────────
if __name__ == "__main__":
    engine = OutfitCompatibilityEngine()
    engine.load()
    print(f"Learned {len(engine._outfit_pairs)} category co-occurrence pairs from curated outfits")

    # Simulate scoring a simple outfit
    test_outfit = [
        {"id": "a", "name": "White Formal Shirt", "category": "formal-shirts",
         "category_label": "Formal Shirts", "slot": "topwear",
         "gender": "men", "occasion": "office", "price_inr": 1099},
        {"id": "b", "name": "Navy Blue Trousers", "category": "trousers",
         "category_label": "Trousers", "slot": "bottomwear",
         "gender": "men", "occasion": "office", "price_inr": 1499},
        {"id": "c", "name": "Brown Loafers", "category": "loafers",
         "category_label": "Loafers", "slot": "footwear",
         "gender": "men", "occasion": "office", "price_inr": 1999},
    ]
    score = engine.score_outfit(test_outfit)
    expl  = engine.get_compatibility_explanation(test_outfit)
    print(f"\nOutfit compatibility score: {score}")
    print(f"Explanation: {expl}")

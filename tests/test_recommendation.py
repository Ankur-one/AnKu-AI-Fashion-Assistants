# ============================================================
# test_recommendation.py — Unit & integration tests
# ============================================================

import sys
import unittest
import numpy as np
from pathlib import Path

# Ensure project root is in path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backend.data_loader import DataLoader
from backend.recommendation.compatibility_engine import OutfitCompatibilityEngine


class TestDataLoader(unittest.TestCase):
    """Tests for the DataLoader class."""

    @classmethod
    def setUpClass(cls):
        cls.loader = DataLoader()
        cls.products = cls.loader.load_products()
        cls.outfits  = cls.loader.load_outfits()

    def test_products_loaded(self):
        self.assertGreater(len(self.products), 0, "Products should not be empty")

    def test_outfits_loaded(self):
        self.assertGreater(len(self.outfits), 0, "Outfits should not be empty")

    def test_products_have_required_columns(self):
        required = ["id", "name", "gender", "category", "occasion", "price_inr", "slot"]
        for col in required:
            self.assertIn(col, self.products.columns, f"Column '{col}' missing")

    def test_slot_assignment(self):
        slots = self.products["slot"].unique().tolist()
        self.assertTrue(any(s in slots for s in ["topwear", "bottomwear", "footwear"]),
                        "At least one clothing slot should exist")

    def test_rich_text_generated(self):
        self.assertIn("rich_text", self.products.columns)
        self.assertTrue(all(self.products["rich_text"].str.len() > 0),
                        "All products should have rich_text")

    def test_get_product_by_id(self):
        first_id = self.products["id"].iloc[0]
        product = self.loader.get_product_by_id(first_id)
        self.assertIsNotNone(product)
        self.assertEqual(product["id"], first_id)

    def test_get_products_by_slot(self):
        topwear = self.loader.get_products_by_slot("topwear")
        self.assertGreater(len(topwear), 0, "Should have some topwear products")

    def test_search_products_gender_filter(self):
        men_products = self.loader.search_products(gender="men")
        if len(men_products) > 0:
            self.assertTrue(all(men_products["gender"].str.lower() == "men"),
                            "All filtered products should be for men")

    def test_dataset_stats(self):
        stats = self.loader.get_dataset_stats()
        self.assertIn("total_products", stats)
        self.assertIn("total_outfits", stats)
        self.assertGreater(stats["total_products"], 0)

    def test_get_outfit_items(self):
        outfit = self.outfits.iloc[0]
        items = self.loader.get_outfit_items(outfit)
        self.assertIn("hero", items)


class TestCompatibilityEngine(unittest.TestCase):
    """Tests for the compatibility scoring engine."""

    @classmethod
    def setUpClass(cls):
        cls.engine = OutfitCompatibilityEngine()
        cls.engine.load()
        cls.good_outfit = [
            {"id": "a", "name": "White Formal Shirt", "category": "formal-shirts",
             "category_label": "Formal Shirts", "slot": "topwear",
             "gender": "men", "occasion": "office", "price_inr": 1099},
            {"id": "b", "name": "Navy Blue Trousers", "category": "trousers",
             "category_label": "Trousers", "slot": "bottomwear",
             "gender": "men", "occasion": "office", "price_inr": 1499},
            {"id": "c", "name": "Brown Leather Loafers", "category": "loafers",
             "category_label": "Loafers", "slot": "footwear",
             "gender": "men", "occasion": "office", "price_inr": 1999},
        ]
        cls.bad_outfit = [
            {"id": "x", "name": "Sports Shorts", "category": "shorts",
             "slot": "bottomwear", "gender": "men", "occasion": "outdoor", "price_inr": 500},
            {"id": "y", "name": "Evening Gown", "category": "dresses",
             "slot": "topwear", "gender": "women", "occasion": "party", "price_inr": 8000},
        ]

    def test_score_range(self):
        score = self.engine.score_outfit(self.good_outfit)
        self.assertGreaterEqual(score, 0.0, "Score must be >= 0")
        self.assertLessEqual(score, 1.0, "Score must be <= 1")

    def test_good_outfit_scores_higher_than_bad(self):
        good_score = self.engine.score_outfit(self.good_outfit)
        bad_score  = self.engine.score_outfit(self.bad_outfit)
        self.assertGreater(good_score, bad_score,
                           "Well-matched outfit should score higher")

    def test_single_item_outfit(self):
        score = self.engine.score_outfit([self.good_outfit[0]])
        self.assertGreaterEqual(score, 0.0)

    def test_empty_outfit(self):
        score = self.engine.score_outfit([])
        self.assertEqual(score, 0.0)

    def test_explanation_generated(self):
        explanation = self.engine.get_compatibility_explanation(self.good_outfit)
        self.assertIsInstance(explanation, str)
        self.assertGreater(len(explanation), 10)

    def test_pair_scoring(self):
        score = self.engine.score_pair(self.good_outfit[0], self.good_outfit[1])
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_gender_consistency_penalised(self):
        mixed_gender = [
            {"id": "m", "category": "formal-shirts", "slot": "topwear",
             "gender": "men", "occasion": "office", "price_inr": 1000},
            {"id": "f", "category": "skirts", "slot": "bottomwear",
             "gender": "women", "occasion": "office", "price_inr": 800},
        ]
        single_gender = [self.good_outfit[0], self.good_outfit[1]]
        mixed_score  = self.engine._score_gender_consistency(mixed_gender)
        single_score = self.engine._score_gender_consistency(single_gender)
        self.assertLess(mixed_score, single_score,
                        "Mixed gender outfit should score lower on gender consistency")

    def test_curated_pairs_learned(self):
        self.assertGreater(len(self.engine._outfit_pairs), 0,
                           "Should have learned some co-occurrence pairs")


class TestSlotMapping(unittest.TestCase):
    """Tests for category → slot assignment."""

    def setUp(self):
        self.loader = DataLoader()

    def test_shirt_is_topwear(self):
        self.assertEqual(self.loader.get_slot("formal-shirts"), "topwear")

    def test_jeans_is_bottomwear(self):
        self.assertEqual(self.loader.get_slot("jeans"), "bottomwear")

    def test_sneakers_is_footwear(self):
        self.assertEqual(self.loader.get_slot("sneakers"), "footwear")

    def test_handbag_is_accessory(self):
        self.assertEqual(self.loader.get_slot("handbags"), "accessory")


class TestEmbedderFallback(unittest.TestCase):
    """Lightweight test that doesn't require actual model loading."""

    def test_l2_normalize(self):
        from backend.embeddings.fashion_clip_embeddings import FashionCLIPEmbedder
        embedder = FashionCLIPEmbedder()
        vecs = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 1.0]])
        normed = embedder._l2_normalize(vecs)
        # First vector: norm should be 1
        self.assertAlmostEqual(np.linalg.norm(normed[0]), 1.0, places=5)
        # Zero vector should not produce NaN
        self.assertFalse(np.any(np.isnan(normed)))

    def test_cosine_similarity(self):
        from backend.embeddings.fashion_clip_embeddings import FashionCLIPEmbedder
        embedder = FashionCLIPEmbedder()
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        sim = embedder.cosine_similarity(a, b)
        self.assertAlmostEqual(sim, 0.0, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)

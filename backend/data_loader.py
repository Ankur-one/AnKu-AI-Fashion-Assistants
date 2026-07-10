# ============================================================
# data_loader.py — Enhanced data loading and preprocessing
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import os

# Allow running standalone
sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.config import (
    PRODUCTS_CSV, OUTFITS_CSV, CURATED_XLSX, IMAGE_DIR,
    SLOT_MAP, TOPWEAR_CATEGORIES, BOTTOMWEAR_CATEGORIES,
    FOOTWEAR_CATEGORIES, ACCESSORY_CATEGORIES
)


class DataLoader:
    """Loads and preprocesses the fashion dataset."""

    def __init__(self):
        self._products: Optional[pd.DataFrame] = None
        self._outfits:  Optional[pd.DataFrame] = None

    # ── Public API ───────────────────────────────────────────

    def load_products(self, force_reload: bool = False) -> pd.DataFrame:
        """Load and preprocess products.csv with caching."""
        if self._products is not None and not force_reload:
            return self._products
        df = pd.read_csv(PRODUCTS_CSV)
        self._products = self._preprocess_products(df)
        return self._products

    def load_outfits(self, force_reload: bool = False) -> pd.DataFrame:
        """Load and preprocess outfits.csv with caching."""
        if self._outfits is not None and not force_reload:
            return self._outfits
        df = pd.read_csv(OUTFITS_CSV)
        self._outfits = self._preprocess_outfits(df)
        return self._outfits

    def load_curated_outfits(self) -> pd.DataFrame:
        """Load the curated25 Excel file."""
        return pd.read_excel(CURATED_XLSX)

    def get_product_by_id(self, product_id: str) -> Optional[pd.Series]:
        """Fetch a single product by its ID."""
        products = self.load_products()
        matches = products[products["id"] == product_id]
        return matches.iloc[0] if not matches.empty else None

    def get_outfit_items(self, outfit_row: pd.Series) -> Dict[str, Optional[pd.Series]]:
        """Resolve an outfit row into its constituent product records."""
        products = self.load_products()
        id_fields = {
            "hero":        "hero_id",
            "second":      "second_id",
            "layer":       "layer_id",
            "footwear":    "footwear_id",
            "accessory_1": "accessory_1_id",
            "accessory_2": "accessory_2_id",
        }
        result = {}
        for slot, id_col in id_fields.items():
            pid = outfit_row.get(id_col)
            if pd.notna(pid) and pid:
                matches = products[products["id"] == pid]
                result[slot] = matches.iloc[0] if not matches.empty else None
            else:
                result[slot] = None
        return result

    def get_slot(self, category: str) -> str:
        """Determine which outfit slot a product category belongs to."""
        cat = category.lower().strip()
        for slot, cats in SLOT_MAP.items():
            if cat in cats or any(c in cat for c in cats):
                return slot
        return "other"

    def get_products_by_slot(self, slot: str) -> pd.DataFrame:
        """Return all products belonging to a given outfit slot."""
        products = self.load_products()
        return products[products["slot"] == slot]

    def search_products(
        self,
        gender: Optional[str] = None,
        occasion: Optional[str] = None,
        slot: Optional[str] = None,
        max_price: Optional[float] = None,
        query: Optional[str] = None,
    ) -> pd.DataFrame:
        """Filter products by metadata criteria."""
        df = self.load_products().copy()
        if gender:
            df = df[df["gender"].str.lower() == gender.lower()]
        if occasion:
            df = df[df["occasion"].str.lower().str.contains(occasion.lower(), na=False)]
        if slot:
            df = df[df["slot"] == slot]
        if max_price is not None:
            df = df[df["price_inr"] <= max_price]
        if query:
            q = query.lower()
            mask = (
                df["name"].str.lower().str.contains(q, na=False) |
                df["description"].str.lower().str.contains(q, na=False) |
                df["category_label"].str.lower().str.contains(q, na=False)
            )
            df = df[mask]
        return df

    def get_image_path(self, relative_path: str) -> Optional[Path]:
        """Resolve a relative image path to an absolute Path."""
        # relative_path looks like "images/myntra/28569210.jpg"
        # We need to look in data/
        from backend.config import DATA_DIR
        full = DATA_DIR / relative_path
        if full.exists():
            return full
        # Try stripping the leading "images/"
        parts = relative_path.split("/")
        if parts[0] == "images" and len(parts) == 3:
            alt = IMAGE_DIR / parts[1] / parts[2]
            if alt.exists():
                return alt
        return None

    def get_dataset_stats(self) -> Dict:
        """Return summary statistics of the dataset."""
        products = self.load_products()
        outfits  = self.load_outfits()
        return {
            "total_products":      len(products),
            "total_outfits":       len(outfits),
            "gender_counts":       products["gender"].value_counts().to_dict(),
            "occasion_counts":     products["occasion"].value_counts().to_dict(),
            "slot_counts":         products["slot"].value_counts().to_dict(),
            "category_counts":     products["category"].value_counts().to_dict(),
            "site_counts":         products["site"].value_counts().to_dict(),
            "avg_price_inr":       round(products["price_inr"].mean(), 2),
            "images_available":    int(products["image_exists"].sum()),
        }

    # ── Private helpers ──────────────────────────────────────

    def _preprocess_products(self, df: pd.DataFrame) -> pd.DataFrame:
        # Normalize strings
        str_cols = ["name", "brand", "gender", "category", "occasion", "description"]
        for col in str_cols:
            if col in df.columns:
                df[col] = df[col].fillna("").str.strip()

        # Clean numeric columns
        df["price_inr"]    = pd.to_numeric(df["price_inr"],    errors="coerce").fillna(0)
        df["rating"]       = pd.to_numeric(df["rating"],       errors="coerce").fillna(0)
        df["rating_count"] = pd.to_numeric(df["rating_count"], errors="coerce").fillna(0)

        # Assign outfit slot
        df["slot"] = df["category"].apply(self.get_slot)

        # Build rich text description for embeddings
        df["rich_text"] = df.apply(self._build_rich_text, axis=1)

        # Validate image existence
        df["image_exists"] = df["image"].apply(
            lambda p: (IMAGE_DIR.parent / p).exists() if p else False
        )

        return df.reset_index(drop=True)

    def _build_rich_text(self, row: pd.Series) -> str:
        """Compose a rich text string for CLIP text embedding."""
        parts = []
        if row.get("name"):
            parts.append(row["name"])
        if row.get("category_label"):
            parts.append(row["category_label"])
        if row.get("occasion"):
            parts.append(f"for {row['occasion']}")
        if row.get("gender"):
            parts.append(f"for {row['gender']}")
        if row.get("description"):
            parts.append(row["description"][:200])  # truncate long descs
        return ". ".join(parts)

    def _preprocess_outfits(self, df: pd.DataFrame) -> pd.DataFrame:
        str_cols = ["outfit_id", "gender", "occasion", "theme",
                    "stylist_rationale", "palette"]
        for col in str_cols:
            if col in df.columns:
                df[col] = df[col].fillna("").str.strip()
        df["items_count"]     = pd.to_numeric(df.get("items_count",     pd.Series()), errors="coerce").fillna(0)
        df["total_price_inr"] = pd.to_numeric(df.get("total_price_inr", pd.Series()), errors="coerce").fillna(0)
        return df.reset_index(drop=True)


# ── Standalone test ──────────────────────────────────────────
if __name__ == "__main__":
    loader = DataLoader()

    products = loader.load_products()
    outfits  = loader.load_outfits()

    print(f"Products : {products.shape}")
    print(f"Outfits  : {outfits.shape}")
    print(f"\nProduct columns:\n{products.columns.tolist()}")
    print(f"\nSlot distribution:\n{products['slot'].value_counts()}")
    print(f"\nGender distribution:\n{products['gender'].value_counts()}")
    print(f"\nOccasion distribution:\n{products['occasion'].value_counts()}")
    print(f"\nImages available: {products['image_exists'].sum()} / {len(products)}")

    stats = loader.get_dataset_stats()
    print(f"\nDataset stats: {stats}")
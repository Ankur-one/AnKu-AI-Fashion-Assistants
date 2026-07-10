# ============================================================
# faiss_index.py — FAISS vector store for similarity search
# ============================================================
# Builds and manages a FAISS IndexFlatIP (inner product) index
# over L2-normalised CLIP embeddings. Since vectors are unit-
# normalised, inner product == cosine similarity.
# ============================================================

import numpy as np
import faiss
import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.config import (
    FAISS_INDEX_PATH, PRODUCT_META_PATH, EMBED_DIM,
    TOP_K_CANDIDATES, VECTOR_DIR
)
from backend.embeddings.fashion_clip_embeddings import FashionCLIPEmbedder


class FashionFAISSIndex:
    """
    Thin wrapper around a FAISS IndexFlatIP that:
    - Builds / persists the index from product embeddings
    - Loads the index on demand (with caching)
    - Supports filtered search (by slot / gender / occasion / price)
    """

    def __init__(self):
        self.index:     Optional[faiss.IndexFlatIP] = None
        self.meta_list: Optional[List[dict]]        = None
        self.embeddings: Optional[np.ndarray]       = None
        self._loaded    = False
        self.embedder   = FashionCLIPEmbedder()

    # ── Build / load ─────────────────────────────────────────

    def build_index(self, force_rebuild: bool = False) -> None:
        """
        Construct the FAISS index from product embeddings.
        On first run this triggers CLIP encoding (slow); subsequent
        runs load from disk (fast).
        """
        embeddings, meta_list = self.embedder.build_product_embeddings(
            force_rebuild=force_rebuild
        )
        self._build_from_arrays(embeddings, meta_list)
        self._save()

    def load_index(self) -> None:
        """Load a previously built index from disk."""
        if self._loaded:
            return
        if not FAISS_INDEX_PATH.exists():
            print("[FAISS] Index not found -- building ...")
            self.build_index()
            return
        print("[FAISS] Loading index from disk ...")
        self.index = faiss.read_index(str(FAISS_INDEX_PATH))
        with open(PRODUCT_META_PATH, "rb") as f:
            self.meta_list = pickle.load(f)
        # Re-reconstruct embedding matrix from index for filtered search
        n = self.index.ntotal
        self.embeddings = np.zeros((n, EMBED_DIM), dtype=np.float32)
        for i in range(n):
            self.embeddings[i] = faiss.rev_swig_ptr(
                self.index.get_xb(), i * EMBED_DIM, EMBED_DIM
            ) if hasattr(self.index, "get_xb") else np.zeros(EMBED_DIM)
        # Faster: just reload from npy if available
        from backend.config import EMBEDDINGS_NPY_PATH
        if EMBEDDINGS_NPY_PATH.exists():
            self.embeddings = np.load(str(EMBEDDINGS_NPY_PATH))
        self._loaded = True
        print(f"[FAISS] Loaded index with {self.index.ntotal} vectors OK")

    # ── Search ───────────────────────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray,
        top_k:           int = TOP_K_CANDIDATES,
        filters:         Optional[Dict[str, Any]] = None,
    ) -> List[Dict]:
        """
        Search for the most similar products to `query_embedding`.

        Args:
            query_embedding: L2-normalised (512,) vector.
            top_k:           Number of results to return.
            filters:         Dict with optional keys:
                               gender, slot, occasion, max_price

        Returns:
            List of product metadata dicts with added 'score' key,
            sorted by descending similarity.
        """
        self._ensure_loaded()

        q = query_embedding.astype(np.float32).reshape(1, -1)

        # ── Apply pre-filters ────────────────────────────────
        candidate_indices = self._apply_filters(filters)

        if not candidate_indices:
            return []

        # Sub-index search: build a small temporary index from candidates
        if len(candidate_indices) <= top_k:
            top_k = len(candidate_indices)

        candidate_embeds = self.embeddings[candidate_indices]   # (M, 512)
        scores = (candidate_embeds @ q.T).flatten()             # (M,)

        # Get top_k indices within the candidate set
        top_local = np.argsort(scores)[::-1][:top_k]
        results = []
        for local_idx in top_local:
            global_idx = candidate_indices[local_idx]
            meta       = dict(self.meta_list[global_idx])
            meta["score"] = float(scores[local_idx])
            results.append(meta)

        return results

    def search_by_text(
        self,
        query_text: str,
        top_k:      int = TOP_K_CANDIDATES,
        filters:    Optional[Dict[str, Any]] = None,
    ) -> List[Dict]:
        """Encode a text query then search."""
        self.embedder.load_model()
        q_embed = self.embedder.encode_query(query_text)
        return self.search(q_embed, top_k=top_k, filters=filters)

    def search_by_product_id(
        self,
        product_id: str,
        top_k:      int = TOP_K_CANDIDATES,
        filters:    Optional[Dict[str, Any]] = None,
        exclude_self: bool = True,
    ) -> List[Dict]:
        """Find products similar to a given product (by ID)."""
        self._ensure_loaded()
        idx = self._get_index_by_id(product_id)
        if idx is None:
            return []
        q_embed = self.embeddings[idx]
        results = self.search(q_embed, top_k=top_k + 1, filters=filters)
        if exclude_self:
            results = [r for r in results if r["id"] != product_id]
        return results[:top_k]

    # ── Slot-based helpers ───────────────────────────────────

    def search_slot(
        self,
        query_embedding: np.ndarray,
        slot:            str,
        gender:          Optional[str] = None,
        occasion:        Optional[str] = None,
        max_price:       Optional[float] = None,
        top_k:           int = 5,
    ) -> List[Dict]:
        """Search within a specific outfit slot (topwear / bottomwear / etc.)."""
        filters = {"slot": slot}
        if gender:
            filters["gender"] = gender
        if occasion:
            filters["occasion"] = occasion
        if max_price is not None:
            filters["max_price"] = max_price
        return self.search(query_embedding, top_k=top_k, filters=filters)

    # ── Utility ──────────────────────────────────────────────

    def get_all_meta(self) -> List[dict]:
        self._ensure_loaded()
        return self.meta_list

    def get_product_embedding(self, product_id: str) -> Optional[np.ndarray]:
        self._ensure_loaded()
        idx = self._get_index_by_id(product_id)
        return self.embeddings[idx] if idx is not None else None

    # ── Private ──────────────────────────────────────────────

    def _build_from_arrays(self, embeddings: np.ndarray, meta_list: List[dict]):
        n = embeddings.shape[0]
        self.index      = faiss.IndexFlatIP(EMBED_DIM)
        self.index.add(embeddings.astype(np.float32))
        self.meta_list  = meta_list
        self.embeddings = embeddings.astype(np.float32)
        self._loaded    = True
        print(f"[FAISS] Built index with {n} vectors OK")

    def _save(self):
        VECTOR_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(FAISS_INDEX_PATH))
        print(f"[FAISS] Saved index -> {FAISS_INDEX_PATH}")

    def _ensure_loaded(self):
        if not self._loaded:
            self.load_index()

    def _get_index_by_id(self, product_id: str) -> Optional[int]:
        for i, m in enumerate(self.meta_list):
            if m["id"] == product_id:
                return i
        return None

    def _apply_filters(self, filters: Optional[Dict[str, Any]]) -> List[int]:
        """Return list of global indices that pass the filter."""
        if not filters:
            return list(range(len(self.meta_list)))

        indices = []
        for i, meta in enumerate(self.meta_list):
            if not self._passes_filter(meta, filters):
                continue
            indices.append(i)
        return indices

    def _passes_filter(self, meta: dict, filters: dict) -> bool:
        # Gender filter (exact or 'unisex' passthrough)
        if "gender" in filters and filters["gender"]:
            g = meta.get("gender", "").lower()
            filt_g = filters["gender"].lower()
            if g and g not in (filt_g, "unisex", ""):
                return False

        # Slot filter
        if "slot" in filters and filters["slot"]:
            if meta.get("slot", "") != filters["slot"]:
                return False

        # Occasion filter (partial match)
        if "occasion" in filters and filters["occasion"]:
            occ = meta.get("occasion", "").lower()
            filt_occ = filters["occasion"].lower()
            if occ and filt_occ not in occ:
                return False

        # Price filter
        if "max_price" in filters and filters["max_price"] is not None:
            price = meta.get("price_inr", 0)
            if price > filters["max_price"]:
                return False

        # Items mentioned filter
        if "items_mentioned" in filters and filters["items_mentioned"]:
            item_matches = False
            target_str = (meta.get("name", "") + " " + meta.get("category_label", "")).lower()
            for item in filters["items_mentioned"]:
                if item.lower() in target_str:
                    item_matches = True
                    break
            if not item_matches:
                return False

        return True


# ── Standalone test ──────────────────────────────────────────
if __name__ == "__main__":
    findex = FashionFAISSIndex()
    findex.build_index(force_rebuild=True)

    # Test text search
    results = findex.search_by_text(
        "white formal shirt for office",
        top_k=5,
        filters={"gender": "men"},
    )
    print("\nSearch: 'white formal shirt for office' (men)")
    for r in results:
        print(f"  [{r['score']:.3f}] {r['name']} — {r['category_label']} — ₹{r['price_inr']}")

    # Test slot search
    results = findex.search_by_text(
        "casual jeans",
        top_k=5,
        filters={"slot": "bottomwear"},
    )
    print("\nSlot search: bottomwear — 'casual jeans'")
    for r in results:
        print(f"  [{r['score']:.3f}] {r['name']} — {r['slot']}")

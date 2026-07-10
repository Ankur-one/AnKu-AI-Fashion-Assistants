# ============================================================
# fashion_clip_embeddings.py — Multi-modal CLIP embeddings
# ============================================================
# Uses open_clip (already installed: open-clip-torch) which is
# version-stable and does NOT have the transformers v5 breakage.
# Falls back to transformers CLIPModel if open_clip unavailable.
# ============================================================

import os
import numpy as np
import pickle
from pathlib import Path
from typing import List, Optional, Tuple
import torch
from PIL import Image
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.config import (
    EMBED_DIM, TEXT_WEIGHT, IMAGE_WEIGHT,
    EMBEDDINGS_NPY_PATH, PRODUCT_META_PATH, MODEL_DIR, DATA_DIR, IMAGE_DIR
)
from backend.data_loader import DataLoader

# ── Model selection ───────────────────────────────────────────
# Use open_clip (stable, no transformers v5 issues)
try:
    import open_clip
    BACKEND = "open_clip"
except ImportError:
    BACKEND = "transformers"

OPEN_CLIP_MODEL  = "ViT-B-32"
OPEN_CLIP_PRETRAIN = "openai"
HF_CLIP_MODEL    = "openai/clip-vit-base-patch32"


class FashionCLIPEmbedder:
    """
    Generates and manages multi-modal CLIP embeddings for fashion products.

    Uses open_clip (ViT-B-32 / OpenAI weights) for stable cross-version support.
    Each product embedding is a weighted fusion of:
      - Text embedding  (0.40): encodes name, category, occasion, description
      - Image embedding (0.60): encodes the product photograph
    Embeddings are L2-normalised so cosine similarity == dot product.
    """

    def __init__(self):
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.model     = None
        self.tokenizer = None
        self.preprocess = None
        self._loaded   = False
        self._backend  = BACKEND

    # ── Model management ─────────────────────────────────────

    def load_model(self):
        """Lazy-load the CLIP model."""
        if self._loaded:
            return

        if self._backend == "open_clip":
            self._load_open_clip()
        else:
            self._load_hf_clip()

    def _load_open_clip(self):
        """Load via open_clip library (stable, recommended)."""
        import warnings, os
        # Suppress harmless Windows symlink + QuickGELU warnings
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        warnings.filterwarnings("ignore", category=UserWarning, module="open_clip")
        warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
        print(f"[CLIP] Loading open_clip {OPEN_CLIP_MODEL} on {self.device} ...")
        import open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            OPEN_CLIP_MODEL,
            pretrained=OPEN_CLIP_PRETRAIN,
            device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer(OPEN_CLIP_MODEL)
        self.model.eval()
        self._loaded = True
        print("[CLIP] Model loaded OK (open_clip backend)")


    def _load_hf_clip(self):
        """Fallback: load via HuggingFace transformers."""
        print(f"[CLIP] Loading HF CLIP {HF_CLIP_MODEL} on {self.device} ...")
        from transformers import CLIPModel, CLIPProcessor
        self.model     = CLIPModel.from_pretrained(HF_CLIP_MODEL).to(self.device)
        self.preprocess = CLIPProcessor.from_pretrained(HF_CLIP_MODEL)
        self.model.eval()
        self._loaded = True
        print("[CLIP] Model loaded OK (transformers backend)")

    # ── Core encode methods ──────────────────────────────────

    def encode_text(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode text strings into L2-normalised 512-dim vectors. Returns (N, 512)."""
        self.load_model()
        if self._backend == "open_clip":
            return self._encode_text_open_clip(texts, batch_size)
        return self._encode_text_hf(texts, batch_size)

    def encode_images(self, image_paths: List[Optional[Path]], batch_size: int = 16) -> np.ndarray:
        """Encode image paths into L2-normalised 512-dim vectors. Returns (N, 512)."""
        self.load_model()
        if self._backend == "open_clip":
            return self._encode_images_open_clip(image_paths, batch_size)
        return self._encode_images_hf(image_paths, batch_size)

    # ── open_clip implementations ─────────────────────────────

    def _encode_text_open_clip(self, texts: List[str], batch_size: int) -> np.ndarray:
        import open_clip
        all_embeds = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            tokens = self.tokenizer(batch).to(self.device)
            with torch.no_grad():
                feats = self.model.encode_text(tokens)   # returns raw tensor always
            feats = feats.cpu().numpy().astype(np.float32)
            all_embeds.append(feats)
        return self._l2_normalize(np.vstack(all_embeds))

    def _encode_images_open_clip(self, image_paths: List[Optional[Path]], batch_size: int) -> np.ndarray:
        all_embeds = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i: i + batch_size]
            images, valid_mask = [], []
            for p in batch_paths:
                try:
                    img = self.preprocess(Image.open(p).convert("RGB")) if p and Path(p).exists() else None
                    images.append(img)
                    valid_mask.append(img is not None)
                except Exception:
                    images.append(None)
                    valid_mask.append(False)

            batch_embed = np.zeros((len(batch_paths), EMBED_DIM), dtype=np.float32)
            valid_images = [img for img, v in zip(images, valid_mask) if v]
            if valid_images:
                img_tensor = torch.stack(valid_images).to(self.device)
                with torch.no_grad():
                    feats = self.model.encode_image(img_tensor)  # raw tensor always
                feats = feats.cpu().numpy().astype(np.float32)
                feats = self._l2_normalize(feats)
                vi = 0
                for j, v in enumerate(valid_mask):
                    if v:
                        batch_embed[j] = feats[vi]
                        vi += 1

            all_embeds.append(batch_embed)
        return np.vstack(all_embeds)

    # ── HuggingFace transformers fallback ─────────────────────
    # (handles BOTH v4 and v5 via sub-model access pattern)

    def _encode_text_hf(self, texts: List[str], batch_size: int) -> np.ndarray:
        all_embeds = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            inputs = self.preprocess(
                text=batch, return_tensors="pt", padding=True, truncation=True, max_length=77
            ).to(self.device)
            with torch.no_grad():
                # Direct sub-model call — stable across HF v4 and v5
                text_out = self.model.text_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                )
                feats = self.model.text_projection(text_out.pooler_output)
            all_embeds.append(feats.cpu().numpy().astype(np.float32))
        return self._l2_normalize(np.vstack(all_embeds))

    def _encode_images_hf(self, image_paths: List[Optional[Path]], batch_size: int) -> np.ndarray:
        all_embeds = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i: i + batch_size]
            images, valid_mask = [], []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB") if p and Path(p).exists() else None
                    images.append(img)
                    valid_mask.append(img is not None)
                except Exception:
                    images.append(None)
                    valid_mask.append(False)

            batch_embed = np.zeros((len(batch_paths), EMBED_DIM), dtype=np.float32)
            valid_images = [img for img, v in zip(images, valid_mask) if v]
            if valid_images:
                inputs = self.preprocess(images=valid_images, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    vision_out = self.model.vision_model(pixel_values=inputs["pixel_values"])
                    feats = self.model.visual_projection(vision_out.pooler_output)
                feats = feats.cpu().numpy().astype(np.float32)
                feats = self._l2_normalize(feats)
                vi = 0
                for j, v in enumerate(valid_mask):
                    if v:
                        batch_embed[j] = feats[vi]
                        vi += 1
            all_embeds.append(batch_embed)
        return np.vstack(all_embeds)

    # ── Single-item convenience ───────────────────────────────

    def encode_single_text(self, text: str) -> np.ndarray:
        return self.encode_text([text])[0]

    def encode_single_image(self, image_path: Path) -> np.ndarray:
        return self.encode_images([image_path])[0]

    def encode_query(self, query_text: str) -> np.ndarray:
        """Encode a user's natural-language query for FAISS search."""
        return self.encode_single_text(query_text)

    # ── Fused product embedding ──────────────────────────────

    def build_product_embeddings(
        self, force_rebuild: bool = False
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        Build (or load cached) multi-modal embeddings for all products.
        Returns: (embeddings [N,512], meta_list [N])
        """
        if not force_rebuild and EMBEDDINGS_NPY_PATH.exists() and PRODUCT_META_PATH.exists():
            print("[CLIP] Loading cached embeddings ...")
            embeddings = np.load(str(EMBEDDINGS_NPY_PATH))
            with open(PRODUCT_META_PATH, "rb") as f:
                meta_list = pickle.load(f)
            print(f"[CLIP] Loaded {len(meta_list)} cached embeddings OK")
            return embeddings, meta_list

        loader   = DataLoader()
        products = loader.load_products()
        n        = len(products)
        print(f"[CLIP] Building embeddings for {n} products ...")

        # 1. Text embeddings
        texts = products["rich_text"].tolist()
        print("[CLIP] Encoding text ...")
        text_embeds = np.zeros((n, EMBED_DIM), dtype=np.float32)
        for i in tqdm(range(0, n, 32), desc="Text batches"):
            batch = texts[i: i + 32]
            text_embeds[i: i + 32] = self.encode_text(batch)

        # 2. Image embeddings
        image_paths = []
        for _, row in products.iterrows():
            rel = row.get("image", "")
            p   = loader.get_image_path(rel) if rel else None
            image_paths.append(p)

        print("[CLIP] Encoding images ...")
        img_embeds = np.zeros((n, EMBED_DIM), dtype=np.float32)
        for i in tqdm(range(0, n, 16), desc="Image batches"):
            batch_paths = image_paths[i: i + 16]
            img_embeds[i: i + 16] = self.encode_images(batch_paths)

        # 3. Fuse: weighted average then re-normalise
        has_image = np.array([p is not None for p in image_paths], dtype=np.float32)
        tw = (TEXT_WEIGHT + (1 - has_image) * IMAGE_WEIGHT)[:, None]
        iw = (IMAGE_WEIGHT * has_image)[:, None]
        fused = self._l2_normalize(tw * text_embeds + iw * img_embeds)

        # 4. Build metadata list
        meta_list = []
        for _, row in products.iterrows():
            meta_list.append({
                "id":             row["id"],
                "name":           row["name"],
                "brand":          row.get("brand", ""),
                "price_inr":      float(row.get("price_inr", 0)),
                "rating":         float(row.get("rating", 0)),
                "gender":         row.get("gender", ""),
                "category":       row.get("category", ""),
                "category_label": row.get("category_label", ""),
                "occasion":       row.get("occasion", ""),
                "slot":           row.get("slot", "other"),
                "description":    row.get("description", ""),
                "image":          row.get("image", ""),
                "site":           row.get("site", ""),
                "product_url":    row.get("product_url", ""),
                "rich_text":      row.get("rich_text", ""),
                "wear_type":      row.get("wear_type", ""),
            })

        # 5. Persist
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        np.save(str(EMBEDDINGS_NPY_PATH), fused)
        with open(PRODUCT_META_PATH, "wb") as f:
            pickle.dump(meta_list, f)
        print(f"[CLIP] Saved embeddings to {EMBEDDINGS_NPY_PATH}")

        return fused, meta_list

    # ── Utility ──────────────────────────────────────────────

    @staticmethod
    def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
        """L2-normalise rows; safe against zero vectors."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1e-8, norms)
        return embeddings / norms

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))


# ── Standalone test ───────────────────────────────────────────
if __name__ == "__main__":
    print(f"Using backend: {BACKEND}")
    embedder = FashionCLIPEmbedder()
    embeddings, meta = embedder.build_product_embeddings(force_rebuild=True)
    print(f"Embeddings shape: {embeddings.shape}")
    q = embedder.encode_query("white formal shirt for office")
    sims = embeddings @ q
    top5 = np.argsort(sims)[::-1][:5]
    print("\nTop-5 similar to 'white formal shirt for office':")
    for idx in top5:
        print(f"  [{sims[idx]:.3f}] {meta[idx]['name']}")

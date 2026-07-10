# Architecture — AI Fashion Outfit Recommendation System

## System Overview

This system is an end-to-end AI-powered fashion recommendation engine that combines **Computer Vision**, **Retrieval-Augmented Generation (RAG)**, and **Conversational AI** to generate complete, explainable outfit recommendations.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         STREAMLIT FRONTEND                          │
│   ┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│   │  Chat Interface  │  │ Product Explorer │  │  Dataset Stats   │  │
│   └────────┬────────┘  └────────┬─────────┘  └──────────────────┘  │
└────────────┼────────────────────┼────────────────────────────────────┘
             │ User Input          │ Product Click
             ▼                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATION LAYER                         │
│                                                                     │
│   ┌─────────────────────────────────────┐                           │
│   │         GeminiFashionChatbot        │                           │
│   │  ┌──────────────────────────────┐   │                           │
│   │  │  Phase A: Intent Extraction  │   │                           │
│   │  │  (Structured JSON from NL)   │   │                           │
│   │  └──────────────┬───────────────┘   │                           │
│   │                 │ {gender, age,      │                           │
│   │                 │  occasion, style,  │                           │
│   │                 │  enriched_query}   │                           │
│   └─────────────────┼───────────────────┘                           │
└───────────────────────────────┼─────────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      RECOMMENDATION ENGINE                          │
│                                                                     │
│   recommend_from_profile()  OR  recommend_from_item()               │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  1. Build enriched query text                               │   │
│   │  2. For each slot (topwear / bottomwear / footwear):        │   │
│   │     └── FAISS search (filtered by gender / occasion / price) │  │
│   │  3. Assemble outfit combinations (cross-product of slots)   │   │
│   │  4. Score each outfit with CompatibilityEngine              │   │
│   │  5. Return top-3 ranked outfits                             │   │
│   └─────────────────────────────────────────────────────────────┘   │
└───────────────────┬─────────────────────┬───────────────────────────┘
                    │                     │
                    ▼                     ▼
┌──────────────────────────┐  ┌──────────────────────────────────────┐
│    FAISS VECTOR STORE     │  │       COMPATIBILITY ENGINE           │
│                          │  │                                      │
│  IndexFlatIP (512-dim)   │  │  Signals:                            │
│  68 product vectors      │  │  • Category slot fitness    (25%)    │
│  Cosine similarity       │  │  • Occasion coherence       (20%)    │
│  Pre-filters:            │  │  • Gender consistency       (15%)    │
│    gender / slot /       │  │  • Color harmony            (15%)    │
│    occasion / price      │  │  • Embedding coherence      (15%)    │
│                          │  │  • Price harmony             (5%)    │
└──────────────────────────┘  │  • Curated co-occurrence     (5%)    │
                              └──────────────────────────────────────┘
                    ▲
                    │ L2-normalised 512-dim vectors
                    │
┌─────────────────────────────────────────────────────────────────────┐
│                     CLIP EMBEDDING LAYER                            │
│                                                                     │
│   FashionCLIPEmbedder  (openai/clip-vit-base-patch32)               │
│                                                                     │
│   ┌──────────────────┐        ┌──────────────────┐                  │
│   │  Text Encoder     │        │  Image Encoder    │                  │
│   │  (Transformer)    │        │  (ViT-B/32)      │                  │
│   │  weight: 0.40     │        │  weight: 0.60     │                  │
│   └────────┬─────────┘        └────────┬──────────┘                  │
│            │ 512-dim                    │ 512-dim                     │
│            └──────────────┬────────────┘                             │
│                           ▼                                          │
│              Weighted fusion → L2 normalise → 512-dim               │
│              (cached in models/embeddings.npy)                       │
└─────────────────────────────────────────────────────────────────────┘
                    ▲
                    │
┌─────────────────────────────────────────────────────────────────────┐
│                          DATA LAYER                                 │
│                                                                     │
│  data/products.csv     68 products (Myntra / Ajio / Nykaa)          │
│  data/outfits.csv      25 curated expert outfits with rationale     │
│  data/images/          68 product photographs (jpg)                 │
│                                                                     │
│  DataLoader → preprocesses, assigns slots, builds rich_text         │
└─────────────────────────────────────────────────────────────────────┘
                                │ Outfits ↑ feed
                    ┌──────────────────────────┐
                    │  Compatibility Learning   │
                    │  Category co-occurrence   │
                    │  from 25 curated outfits  │
                    └──────────────────────────┘

                                │ Recommendations ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        EXPLANATION LAYER                            │
│                                                                     │
│   GeminiFashionChatbot                                              │
│   ┌──────────────────────────────────┐                              │
│   │  Phase B: Explanation Generation │                              │
│   │  Input: outfit items + metadata  │                              │
│   │  Output: Natural language rationale (2-3 sentences)            │  │
│   │  Fallback: rule-based template   │                              │
│   └──────────────────────────────────┘                              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Data Layer (`backend/data_loader.py`)
- Loads `products.csv` and `outfits.csv`
- Preprocesses: normalizes strings, fills nulls, converts price/rating to numeric
- Assigns each product to an **outfit slot** (`topwear`, `bottomwear`, `footwear`, `accessory`) based on category
- Builds a **rich_text** field concatenating name + category + occasion + description for CLIP text encoding
- Validates image path existence

### 2. CLIP Embedding Layer (`backend/embeddings/fashion_clip_embeddings.py`)
- Uses `openai/clip-vit-base-patch32` via HuggingFace Transformers
- **Text embedding**: Encodes the `rich_text` field (name + category + description) → 512-dim
- **Image embedding**: Encodes product photos → 512-dim
- **Fusion**: Weighted average (text=0.40, image=0.60), then L2-normalise
- For missing images: falls back entirely to text embedding
- **Caching**: Saves `models/embeddings.npy` and `vector_db/product_meta.pkl` to avoid re-encoding

### 3. FAISS Vector Store (`backend/embeddings/faiss_index.py`)
- `IndexFlatIP` (inner product) — cosine similarity on L2-normalised vectors
- Supports **pre-filtering** by slot, gender, occasion, max_price before similarity search
- Efficient sub-index search on filtered candidates
- Persists index to `vector_db/fashion.index`

### 4. Compatibility Engine (`backend/recommendation/compatibility_engine.py`)
Multi-signal scoring with weighted combination:

| Signal | Weight | Description |
|--------|--------|-------------|
| Category slot fitness | 25% | Right items in right slots, no duplicates |
| Occasion coherence | 20% | Items share compatible occasions |
| Gender consistency | 15% | All items target same gender |
| Color harmony | 15% | Color theory: neutrals + accent, family grouping |
| Embedding coherence | 15% | CLIP similarity in [0.15, 0.65] sweet spot |
| Price harmony | 5% | Low coefficient of variation in prices |
| Curated co-occurrence | 5% | Category pairs seen in expert-curated outfits |

### 5. Recommendation Engine (`backend/recommendation/recommendation_engine.py`)
Two operating modes:
- **`recommend_from_item(product_id)`**: Fixed seed item + fill remaining slots via FAISS
- **`recommend_from_profile(query, gender, age, occasion, style)`**: Query-driven slot-wise FAISS search → cross-product assembly → compatibility ranking

### 6. Gemini Chatbot (`backend/chatbot/gemini_chatbot.py`)
Three roles:
- **Intent Extraction**: Parses natural language → structured JSON (gender, age, occasion, style, enriched_query)
- **Explanation Generation**: Generates 2-3 sentence styling rationale per outfit
- **Conversational Chat**: Multi-turn dialogue with history, personalized responses

Robust fallbacks for each role when the API key is unavailable.

### 7. Streamlit UI (`frontend/app.py`)
Three tabs:
- **Chat Assistant**: Conversational interface with outfit cards, images, scores
- **Explore Products**: Filtered product browser with "Build outfit" action
- **Dataset Stats**: Distribution charts and dataset overview

---

## Data Flow: End-to-End Example

```
User: "I need an outfit for a business meeting"
         │
         ▼
Gemini Intent Extraction:
  → { gender: null, occasion: "office", style: "formal",
      enriched_query: "business meeting formal wear office" }
         │
         ▼
CLIP encodes enriched_query → 512-dim query vector
         │
         ▼
FAISS slot search (3 parallel queries, filtered by occasion="office"):
  topwear    → [Cotton Slim Fit Formal Shirt, ...]
  bottomwear → [Navy Blue Trousers, ...]
  footwear   → [Brown Oxford Shoes, ...]
         │
         ▼
Outfit assembly (cross-product of top-3 per slot → 27 combinations)
         │
         ▼
Compatibility scoring: each outfit gets 0-1 score
  e.g. Outfit A: 0.87 — formal + formal + formal, occasion=office, navy+white+brown
         │
         ▼
Top-3 outfits returned
         │
         ▼
Gemini Explanation:
  "The white formal shirt's clean cut pairs perfectly with navy trousers —
   a classic combination that projects professionalism. Brown leather Oxford
   shoes complete the look with warm-toned sophistication."
         │
         ▼
Streamlit renders: outfit cards with images, scores, explanation
```

---

## Key Design Decisions

1. **FashionCLIP over TF-IDF**: Multi-modal embeddings capture visual and semantic similarity in a shared space, enabling "search by feel" (not just keyword matching).

2. **FAISS over brute-force**: Even at 68 items it's efficient; scales to thousands with no code changes.

3. **Slot-based search**: Rather than searching the full catalog, search per slot (topwear, bottomwear, footwear) to guarantee complete outfits with the right item types.

4. **Multi-signal compatibility**: Single-metric ranking (just CLIP similarity) would produce redundant outfits (all formal shirts together). Multi-signal scoring produces genuinely compatible combinations.

5. **Curated outfit learning**: The 25 expert outfits teach category co-occurrence patterns, grounding the ML in real fashion expertise.

6. **Graceful degradation**: Every Gemini call has a rule-based fallback, so the system works end-to-end even without an API key.

7. **Embedding fusion (image 60%, text 40%)**: Images are the primary signal in fashion (color, texture, fit); text provides category/occasion context.

---

## Future Improvements

- **Graph-based compatibility**: Model outfit relationships as a graph (GNN) to learn complex multi-item compatibility patterns
- **User feedback loop**: Track which recommendations users click/save → improve ranking via implicit feedback
- **Fine-tuned FashionCLIP**: Fine-tune on Indian fashion dataset for better domain alignment
- **Pairwise ranking model**: Learn to rank outfit pairs based on human preferences
- **Hybrid search**: Combine dense (FAISS) + sparse (BM25) retrieval for better recall on rare items
- **Multi-modal input**: Allow users to upload a photo of an item they own and find compatible pieces

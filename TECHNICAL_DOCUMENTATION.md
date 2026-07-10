# Technical Documentation: DARE XAI

## 1. Backend Architecture
The backend is built in pure Python and is designed around a modular object-oriented architecture. The core components are completely decoupled from the Streamlit frontend.
- **Data Layer (`data_loader.py`)**: Responsible for reading CSV files, handling missing values, and mapping image file paths.
- **Embedding Layer (`fashion_clip_embeddings.py` & `faiss_index.py`)**: Manages the loading of the `open_clip` model, generates 512-dimensional text/image embeddings, and manages the FAISS vector indexing and serialization.
- **NLP Layer (`gemini_chatbot.py`)**: Wraps the `google.generativeai` SDK. Exposes methods for JSON intent extraction and natural language explanation generation. Contains robust fallback regex algorithms for when the LLM is unavailable.
- **Recommendation Layer (`recommendation_engine.py` & `compatibility_engine.py`)**: Contains the business logic for assembling outfits, scoring item compatibility, applying budget constraints, and executing the diversity penalty loop.

## 2. Frontend Architecture
The frontend is a single-page application built using Streamlit (`app.py`). 
- **State Management**: Uses `st.session_state` to persist chat history, loaded models, and the current user profile across reruns.
- **Styling**: Extensive use of injected CSS (`st.markdown(..., unsafe_allow_html=True)`) to override Streamlit's default components, enabling a luxury dark mode, glassmorphism (`backdrop-filter: blur`), and custom typography (Playfair Display and Inter).
- **Component Rendering**: Encapsulates UI elements into discrete functions (`render_outfit_card`, `render_product_grid`, `render_sidebar`).

## 3. Data Flow Diagram

```text
[User Input] --> (Streamlit UI)
                      |
                      v
             [Gemini Chatbot] ---> (Intent Extraction: JSON)
                      |
       (Fallback Regex if API fails)
                      |
                      v
          [Recommendation Engine]
                      |
   +------------------+------------------+
   |                                     |
[Query Route]                     [Outfit Route]
   |                                     |
   v                                     v
(Extract Search Term)         (Extract Occasion, Style, Budget)
   |                                     |
   v                                     v
[CLIP Model] ---> (Query Vector) <--- [CLIP Model]
   |                                     |
   v                                     v
[FAISS Index] ---> (KNN Search) <--- [FAISS Index]
   |                                     |
   v                                     v
(Filter by Budget & Gender)    (Assemble Slots: Top, Bottom, Footwear)
   |                                     |
   v                                     v
[Return Products]             [Rank & Penalize Duplicates]
                                         |
                                         v
                              [Generate LLM Explanations]
                                         |
                                         v
[Streamlit UI Render] <------------------+
```

## 4. API Workflow
1. User submits a query via the Streamlit chat input.
2. `gemini_chatbot.extract_intent()` is called. It sends a prompt to Gemini 2.5 Flash requesting a structured JSON response containing `gender`, `age`, `occasion`, `style`, `max_price`, and `query_intent`.
3. If an HTTP 429 (Quota Exceeded) error is thrown, the `except` block catches it and immediately routes the query to `_rule_based_intent()`.
4. The extracted JSON dictates whether `engine.search_products()` or `engine.recommend_from_profile()` is invoked.

## 5. Recommendation Engine Design
The engine utilizes a **hybrid scoring approach**. 
1. **Semantic Score**: FAISS L2 distance is converted into a similarity percentage using CLIP.
2. **Compatibility Score**: Hard-coded heuristics reward exact matches on the `occasion` and `style` metadata columns.
3. **Diversity Score**: An iterative penalty is applied during outfit assembly. If an item `ID` was used in `Outfit A`, its score is artificially reduced when evaluating `Outfit B`, forcing the engine to select alternative items and promoting catalog diversity.

## 6. Folder Structure Explanation
- `/backend/config.py`: Contains global constants, paths, and API keys.
- `/backend/data_loader.py`: Pandas abstractions for CSV reading.
- `/backend/embeddings/`: Contains the CLIP wrapper and FAISS integration. Keeps heavy ML dependencies isolated.
- `/backend/recommendation/`: The algorithmic core. Contains the routing logic and the scoring math.
- `/backend/chatbot/`: External API integrations (Google Gemini).
- `/frontend/app.py`: The presentation layer.
- `/data/`: Static storage for CSVs, FAISS index binaries, and local images.

## 7. Model Loading Process
To prevent memory exhaustion and long load times, models are loaded exactly once and cached:
- The `@st.cache_resource` decorator in `app.py` ensures that `open_clip` and the FAISS index are only loaded into RAM on the very first user interaction. Subsequent user sessions share the same model instance in memory.

## 8. Vector Database Creation Process
1. On application startup, `faiss_index.py` checks for `data/embeddings.npy`.
2. If missing, it iterates through every row in `products.csv`.
3. It constructs a rich text string combining the product name, brand, description, and category.
4. `open_clip` generates a vector for this string.
5. Vectors are normalized and added to `faiss.IndexFlatL2`.
6. The index is written to disk for future O(1) load times.

## 9. Error Handling Strategy
- **API Failures**: Wrapped in `try-except` blocks. Silent fallbacks take over without leaking stack traces to the UI.
- **Missing Images**: `load_product_image()` catches `FileNotFoundError` and `PIL` exceptions, returning `None`. The UI reacts by rendering a CSS-styled fallback placeholder (`👕`).
- **Data Mismatches**: If FAISS returns an ID that no longer exists in the CSV, the engine ignores it gracefully.

## 10. Performance Optimization Techniques
- **Pre-computed Embeddings**: Text embeddings are generated once and saved to disk.
- **Top-K Truncation**: FAISS only searches the Top 50 nearest neighbors before passing them to the Python-level compatibility engine, saving CPU cycles.
- **Batched Generation**: Fallback explanations are generated via string formatting (which takes microseconds) rather than synchronous blocking API calls when the quota is depleted.

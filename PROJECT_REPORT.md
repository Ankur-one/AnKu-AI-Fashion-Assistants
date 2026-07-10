# Project Report: DARE XAI – AI Fashion Recommendation System

## 1. Abstract
The "DARE XAI" project presents an advanced Artificial Intelligence-driven fashion recommendation system. By leveraging Contrastive Language-Image Pretraining (CLIP) for deep semantic embeddings and Large Language Models (LLMs) via the Gemini API for natural language understanding, the system effectively bridges the gap between raw e-commerce catalogs and personalized styling. The project features a resilient architecture with local fallbacks, high-speed vector retrieval via FAISS, and a premium luxury frontend built on Streamlit.

## 2. Introduction
E-commerce fashion retail heavily relies on effective product discovery. Traditional keyword-based search engines struggle to capture the nuance of fashion semantics, such as "smart casual for a summer date under 3000." DARE XAI resolves this by implementing a hybrid recommendation engine that understands both visual styling cues and complex natural language user queries, acting as an automated personal stylist.

## 3. Problem Statement
Current fashion recommendation systems face several challenges:
- **Poor Contextual Understanding**: Inability to map natural language queries to appropriate product tags.
- **Lack of Cohesion**: Suggesting individual products rather than complete, stylistically matching outfits.
- **Rigid Filtering**: Binary budget filters that fail to suggest slightly out-of-budget but highly relevant alternatives.
- **System Fragility**: Total failure of chatbot interfaces when third-party AI APIs experience downtime or quota exhaustion.

## 4. Objectives
1. Develop an intelligent intent extraction module using LLMs.
2. Build a vector search pipeline to index and retrieve fashion items using CLIP embeddings.
3. Design a cohesive outfit assembly engine that respects occasion, style, gender, and budget constraints.
4. Implement a graceful fallback mechanism to ensure 100% uptime regardless of API limits.
5. Deliver a premium, responsive UI that mimics luxury fashion brands.

## 5. Dataset Description
The system utilizes a custom fashion catalog comprising two primary sources:
- **`products.csv`**: Contains product metadata including `id`, `name`, `category_label`, `price_inr`, `gender`, `occasion`, `style`, `brand`, and `image` file paths.
- **`outfits.csv`**: Contains pre-curated combinations of product IDs to serve as ground-truth stylistic templates.
- **Image Directory**: High-resolution product images corresponding to the metadata.

## 6. Methodology
The project employs a modular, pipeline-driven methodology:
1. **Data Ingestion**: Raw data is sanitized and loaded into pandas DataFrames.
2. **Vectorization**: Product names, descriptions, and category labels are passed through the CLIP text encoder to generate dense vector embeddings.
3. **Indexing**: Embeddings are stored in a FAISS FlatL2 index for O(1) semantic retrieval.
4. **Intent Parsing**: User queries are analyzed by Gemini to extract structured JSON parameters.
5. **Retrieval & Ranking**: FAISS retrieves candidates which are then re-ranked based on price, occasion compatibility, and stylistic diversity.

## 7. System Design
The system architecture consists of a Streamlit frontend communicating with a Python backend. The backend is bifurcated into the Chatbot Module (Handling LLM API calls) and the Recommendation Engine (Handling FAISS retrieval and assembly logic).

### 7.1 CLIP Embedding Workflow
1. The `open_clip` model is loaded into memory.
2. During initialization, textual descriptions of all products are tokenized.
3. The model generates 512-dimensional embeddings for each product.
4. These embeddings are cached to disk to eliminate redundant computation on system restarts.

### 7.2 FAISS Search Workflow
1. The 512-dimensional CLIP embeddings are loaded into a `faiss.IndexFlatL2` structure.
2. When a search occurs, the user's query is vectorized via the same CLIP model.
3. FAISS performs a K-Nearest Neighbors (KNN) search to return the closest product vectors.

### 7.3 Gemini Integration
Google's Gemini 2.5 Flash model is integrated for two specific tasks:
- **Intent Extraction**: Parsing unstructured text into a deterministic JSON schema.
- **Explanation Generation**: Generating rich, natural-language rationales explaining *why* an outfit was recommended.

## 8. Recommendation Pipeline
1. **Filtering**: Pre-filter the catalog based on hard constraints (e.g., exact gender match).
2. **Retrieval**: Use FAISS to fetch the top `N` semantically relevant items for each required clothing slot (Topwear, Bottomwear, Footwear, Accessory).
3. **Assembly**: Iteratively select the highest-scoring items that fit within the remaining budget.
4. **Diversity Penalization**: Apply a penalty score to item IDs that have already been recommended in previous outfits to ensure visual diversity across recommendations.

## 9. Fallback Engine Logic
To address Gemini's `429 Quota Exceeded` errors, a robust fallback strategy is implemented:
- **Rule-Based Intent Extraction**: Regular expressions and keyword mapping are used to deduce age, gender, and budget from the query if the LLM fails.
- **Heuristic Explanation Generation**: Template-based string formatting generates styling rationales using the product's category, name, and occasion metadata.
- **Silent Degradation**: The frontend suppresses the stack trace and presents a premium fallback message to the user, ensuring the user experience remains uninterrupted.

## 10. Results and Testing
The system was rigorously tested across edge cases:
- Queries requesting specific items correctly bypass outfit generation and route to the Product Search pipeline.
- Complex constraints ("date night outfit under 5000") successfully yield complete outfits matching the criteria.
- API failures seamlessly trigger the local engine without UI breakage.

## 11. Challenges Faced
- **API Rate Limits**: Frequent quota exhaustion from Gemini necessitated the development of the complex local fallback engine.
- **Diversity Collapse**: Initial algorithms repeatedly suggested the same topwear item across multiple outfits. This was solved by introducing a dynamic diversity penalty during the ranking phase.
- **UI Rendering Anomalies**: Streamlit's Markdown parser misinterpreted indented HTML as code blocks, requiring careful string deduplication and formatting to render the glassmorphism UI correctly.

## 12. Future Scope
- Transitioning from Text-to-Text CLIP search to Image-to-Image search, allowing users to upload a photo and find visually similar items.
- Implementing a persistent user database to track long-term style preferences and purchase history.
- Migrating the Streamlit frontend to a dedicated React/Next.js application for enterprise-grade scalability.

## 13. Conclusion
DARE XAI successfully demonstrates the immense potential of combining foundational vision-language models (CLIP) with conversational AI (Gemini). By engineering a fault-tolerant backend and a meticulously designed luxury frontend, the project serves as a highly capable prototype for next-generation e-commerce fashion retail.

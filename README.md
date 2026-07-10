# AnKu AI – AI Fashion Recommendation System


##  Project Overview
AnKu AI is a premium, AI-powered personal styling and fashion recommendation engine. It bridges the gap between sophisticated artificial intelligence and luxury fashion by providing users with highly curated, occasion-specific, and budget-aware outfit recommendations. By combining the power of OpenAI's CLIP embeddings for deep visual-semantic understanding and Google's Gemini API for natural language intent extraction, AnKu AI acts as an elite personal stylist. 

##  Features
- **AI-Powered Outfit Recommendation**: Generates complete outfits intelligently matched by style, occasion, and budget.
- **Natural Language Product Search**: Users can search for items organically (e.g., "I need shoes under 1000 for men").
- **Semantic Search via CLIP**: Employs OpenAI's CLIP model to understand the deep semantic relationships between product images and text descriptions.
- **Personalized Outfit Generation**: Tailors recommendations based on user profiles (Age, Gender, Occasion, Style Preference).
- **Budget-Aware Engine**: strict exact-match budget filtering with intelligent fallback suggestions.
- **Resilient Fallback Architecture**: Seamlessly falls back to a high-quality local rule-based engine if the Gemini API quota is exhausted, ensuring zero downtime.
- **Premium Luxury UI**: A visually stunning, typography-first Streamlit interface using glassmorphism and modern UI principles.
- **Dataset Analytics Dashboard**: Provides an interactive overview of catalog distribution, occasions, categories, and pricing.

##  System Architecture
The application follows a modular, decoupled architecture:
1. **Frontend**: Streamlit-based luxury UI.
2. **Intent Engine**: Gemini API processes raw user input to extract structured parameters (budget, occasion, gender, product intent).
3. **Embedding Pipeline**: Products are vectorized using CLIP (`open_clip`) and indexed via FAISS for blazing-fast similarity search.
4. **Recommendation Engine**: A hybrid engine that combines vector similarity scores with heuristic penalties (e.g., diversity scoring) to rank products and assemble cohesive outfits.

##  Tech Stack
- **Backend**: Python 3.10+
- **Frontend**: Streamlit
- **AI/ML Models**: CLIP (Contrastive Language-Image Pretraining), Google Gemini 2.5 Flash
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Data Manipulation**: Pandas, NumPy
- **Image Processing**: Pillow (PIL)

##  Installation Guide

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/dare-xai.git
   cd dare-xai
   ```

2. **Set up a Virtual Environment**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**
   Create a `.env` file in the root directory and add your Gemini API Key:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

##  How To Run

1. Ensure your dataset (`products.csv`, `outfits.csv`, and images) are placed in the `data/` directory.
2. Initialize the FAISS index (if running for the first time, the system will auto-generate it).
3. Start the Streamlit server:
   ```bash
   streamlit run frontend/app.py
   ```
4. Open your browser and navigate to `http://localhost:8501`.

##  Project Structure
```text
dare-xai/
├── backend/
│   ├── chatbot/              # Gemini integration and intent extraction
│   ├── embeddings/           # CLIP model loading and FAISS index generation
│   ├── recommendation/       # Core hybrid recommendation logic and outfit assembly
│   ├── config.py             # System-wide configuration
│   └── data_loader.py        # CSV and Image loading utilities
├── frontend/
│   └── app.py                # Streamlit UI, styling, and rendering logic
├── data/                     # CSV datasets and product images
├── tests/                    # Unit testing and trace debugging scripts
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

Author

Avinash Singh

B.Tech (Artificial Intelligence & Machine Learning)

AI/ML Engineer

LinkedIn: https://linkedin.com/in/avinash-singh-1a5832262

GitHub: https://github.com/avinashsingh0218

## Dataset Analysis Findings

Based on our recent analysis of the `products.csv` and `outfits.csv` datasets, here are the key insights:

* **Catalog Size**: The dataset consists of **68 total products** and **25 curated outfits**.
* **Diversity**: We observed **47 unique categories** across **64 distinct brands**, ensuring a wide variety of recommendations.
* **Category Distribution**: The catalog is well-balanced across topwear, bottomwear, footwear, and accessories, with topwear being the most frequent.
* **Occasions**: The dataset covers 8 unique occasions, with **Casual**, **Party**, and **Office** wear being the most prominent.
* **Pricing**: Products range from ₹270 to ₹7,799, with an average price of ₹1,518 and a median of ₹1,082, catering to various budget requirements.
* **Data Quality**: The core metadata fields (category, price, gender) and product images have 0 missing values, ensuring high reliability for the recommendation engine. The only missing data found was in optional rating fields.
import os

folders = [
    "data",
    "data/images",
    "backend",
    "backend/recommendation",
    "backend/chatbot",
    "backend/embeddings",
    "frontend",
    "vector_db",
    "models",
    "docs",
    "tests"
]

files = [
    "backend/__init__.py",
    "backend/recommendation/__init__.py",
    "backend/chatbot/__init__.py",
    "backend/embeddings/__init__.py",

    "backend/recommendation/recommendation_engine.py",
    "backend/recommendation/compatibility_engine.py",

    "backend/chatbot/gemini_chatbot.py",

    "backend/embeddings/fashion_clip_embeddings.py",
    "backend/embeddings/faiss_index.py",

    "frontend/app.py",

    "vector_db/.gitkeep",
    "models/.gitkeep",

    "docs/architecture.md",

    "tests/test_recommendation.py",

    "requirements.txt",
    "README.md",
    ".env"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

for file in files:
    with open(file, "w") as f:
        pass

print("✅ Project structure created successfully!")         
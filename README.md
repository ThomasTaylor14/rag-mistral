# Mistral RAG Financial Analyst

RAG-powered chatbot that enables natural language querying of BNP Paribas Q2 2025 earnings data, providing instant access to financial metrics and insights from earnings calls and press releases.

## 🚀 Quick Setup

### Option 1: Using `uv` (Recommended - Faster)
```bash
uv sync
```

### Option 2: Using `pip` + `venv`
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 🔑 Environment Setup

1. Create a `.env` file in the project root:
```bash
MISTRAL_API_KEY=your_mistral_api_key_here
```

2. Get your API key from [Mistral AI Console](https://console.mistral.ai/)

## 📊 Prepare the Data

Before running the app, build the vector index by running the Jupyter notebook:

```bash
# Option 1: With uv
uv run jupyter notebook financial_analyst_rag.ipynb

# Option 2: With pip/venv
jupyter notebook financial_analyst_rag.ipynb
```

Run all cells to create the FAISS index from the financial documents.

## 🎯 Run the App

```bash
# Option 1: With uv
uv run streamlit run demo_app.py

# Option 2: With pip/venv  
streamlit run demo_app.py
```

Open your browser to `http://localhost:8501` and start asking questions about BNP Paribas financials!

## 💬 Example Questions

- "What was the Group's net income expected for 2025?"
- "What was the interim dividend per share announced for 2025?"
- "Summarize the bank's 2026 growth trajectory"

## 🛠️ Tech Stack

- **LLM**: Mistral Large
- **Embeddings**: Mistral Embed
- **Vector DB**: FAISS
- **Frontend**: Streamlit
- **Document Processing**: PyMuPDF, LangChain
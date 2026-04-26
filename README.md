# Enterprise Decision Intelligence Platform (EIDAP)

![EIDAP Banner](https://img.shields.io/badge/Status-Deploy_Ready-success) ![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Reflex](https://img.shields.io/badge/Framework-Reflex-indigo)

A premium full-stack, AI-powered enterprise analytics SaaS platform built entirely in Python using **Reflex**. 

This platform leverages the real-world **UCI Online Retail Dataset** to demonstrate end-to-end Machine Learning pipelines including Customer Segmentation, Churn Prediction, Demand Forecasting, Customer Lifetime Value (CLV) scoring, and an interactive Retrieval-Augmented Generation (RAG) AI Assistant.

## Features
- **Customer Intelligence**: KMeans clustering based on Recency, Frequency, and Monetary (RFM) value.
- **Churn Analytics**: Random Forest Classification to predict the likelihood of an account churning.
- **Sales Forecasting**: Facebook Prophet time-series models to forecast 90-day demand.
- **CLV Prediction**: Regression models estimating 12-month future value.
- **AI Business Assistant**: RAG engine powered by `FAISS` and `SentenceTransformers` allowing you to converse with your business data.
- **Responsive Dashboard**: Complete Python-only reactive frontend with multi-page routing and premium glassmorphism UI.

## Architecture & Tech Stack
- **Frontend & Backend**: Reflex (Python)
- **Machine Learning**: Scikit-Learn (Random Forest, KMeans), Prophet (Time-series Forecasting)
- **Vector Search Engine**: FAISS (Local dense vector retrieval)
- **Embeddings**: SentenceTransformers (`all-MiniLM-L6-v2`)

---

## 🚀 Setup Instructions

Follow these steps to run the application natively:

### 1. Install Dependencies
Ensure you have Python 3.10+ installed.
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Data & ML Pipeline
Before starting the application, you must download the real dataset and train the machine learning models. 
```bash
python run_pipelines.py
```
**What this does:**
1. Downloads the **UCI Online Retail Dataset**.
2. Cleans the data and calculates RFM scores.
3. Trains and serializes the Machine Learning models into the `/models` directory.
4. Synthesizes a knowledge base and builds a local `FAISS` vector index in `/data`.

### 3. Start the Reflex Application
Once the pipeline completes, launch the platform:
```bash
reflex run
```

### 4. Access the Dashboard
Navigate to **http://localhost:3000** to explore the platform.

---

## 🌍 Production Deployment

The easiest way to deploy this full-stack application to the internet is using Reflex's native hosting service, or a PaaS like Render.

### Option 1: Reflex Deploy (Easiest)
Reflex provides a managed hosting service that handles both the Python backend and the frontend automatically.
```bash
reflex login
reflex deploy
```

### Option 2: Render (GitHub Integration)
1. Push this repository to GitHub.
2. Log into [Render](https://render.com/) and create a new **Web Service**.
3. Connect your GitHub repository.
4. Set the **Build Command**: `pip install -r requirements.txt && python run_pipelines.py`
5. Set the **Start Command**: `reflex run --env prod`

*(Note: Because this application uses Machine Learning models and FAISS, the deployment environment needs at least 1GB of RAM).*

---

## License
MIT License. Free for portfolio use and modifications.

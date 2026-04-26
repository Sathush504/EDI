# Enterprise Decision Intelligence Platform (EIDAP)

![EIDAP Banner](https://img.shields.io/badge/Status-Live-success) ![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B)

A premium, AI-powered enterprise analytics SaaS platform built entirely in Python using **Streamlit**. 

This platform leverages the real-world **UCI Online Retail Dataset** to demonstrate end-to-end Machine Learning pipelines including Customer Segmentation, Churn Prediction, Demand Forecasting, Customer Lifetime Value (CLV) scoring, and an interactive Retrieval-Augmented Generation (RAG) AI Assistant.

## Features
- **Customer Intelligence**: KMeans clustering based on Recency, Frequency, and Monetary (RFM) value.
- **Churn Analytics**: Random Forest Classification to predict the likelihood of an account churning.
- **Sales Forecasting**: Facebook Prophet time-series models to forecast 90-day demand.
- **CLV Prediction**: Regression models estimating 12-month future value.
- **AI Business Assistant**: RAG engine powered by `FAISS` and `SentenceTransformers` allowing you to converse with your business data.
- **Responsive Dashboard**: Beautiful, multi-page data application using native Streamlit routing and Plotly charting with a custom dark theme.

## Architecture & Tech Stack
- **Frontend & App Framework**: Streamlit
- **Machine Learning**: Scikit-Learn (Random Forest, KMeans), Prophet (Time-series Forecasting)
- **Vector Search Engine**: FAISS (Local dense vector retrieval)
- **Embeddings**: SentenceTransformers (`all-MiniLM-L6-v2`)

---

## 🚀 Running Locally

Follow these steps to run the application on your own machine:

### 1. Install Dependencies
Ensure you have Python 3.10+ installed.
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start the Application
The application is pre-packaged with generated intelligence data so you can launch the dashboard immediately.
```bash
streamlit run streamlit_app.py
```
This will automatically open the dashboard in your browser at `http://localhost:8501`.

*(Optional: If you wish to regenerate the data from scratch or retrain the models, you can run `python run_pipelines.py`)*

---

## 🌍 Production Deployment

This application is architected as a lightweight, single-file Streamlit app, making deployment incredibly simple.

### Deploying to Streamlit Community Cloud (Recommended)
1. Push this repository to your GitHub account.
2. Log into [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **New app**.
4. Select your GitHub repository.
5. Set the **Main file path** to `streamlit_app.py`.
6. Click **Deploy!**

*(Note: Because the AI Assistant loads ML weights into memory, the very first boot of the deployed app may take 15-20 seconds.)*

---

## License
MIT License. Free for portfolio use and modifications.

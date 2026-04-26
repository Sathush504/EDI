import pandas as pd
import numpy as np
import os
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from prophet import Prophet
import warnings

warnings.filterwarnings('ignore')

def train_segmentation(df):
    features = ['recency', 'frequency', 'monetary']
    X = df[features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Labeling based on recency/monetary
    cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=features)
    cluster_centers['score'] = cluster_centers['frequency'] * cluster_centers['monetary'] / (cluster_centers['recency'] + 1)
    
    # Sort clusters by score descending
    sorted_clusters = cluster_centers.sort_values(by='score', ascending=False).index.tolist()
    labels = ["Enterprise Champions", "Core Accounts", "At-Risk Accounts", "Dormant"]
    labels_map = {sorted_clusters[i]: labels[i] for i in range(4)}
    
    df['segment_label'] = df['cluster'].map(labels_map)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.joblib')
    joblib.dump(kmeans, 'models/kmeans.joblib')
    return df

def train_churn(df):
    features = ['recency', 'frequency', 'monetary']
    X = df[features]
    y = df['churned']
    
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)
    
    df['churn_prob'] = model.predict_proba(X)[:, 1]
    
    def get_risk(p):
        if p < 0.3: return 'Low'
        if p < 0.7: return 'Medium'
        return 'High'
        
    df['churn_risk'] = df['churn_prob'].apply(get_risk)
    joblib.dump(model, 'models/churn_rf.joblib')
    return df

def train_clv(df):
    """Predicts future target CLV based on RFM"""
    features = ['recency', 'frequency', 'monetary', 'churn_prob']
    X = df[features]
    y = df['target_clv']
    
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)
    
    df['predicted_clv'] = model.predict(X)
    joblib.dump(model, 'models/clv_model.joblib')
    return df

def generate_recommendations(df):
    """Rule-based next best action"""
    def action(row):
        if row['segment_label'] == "Enterprise Champions":
            return "Cross-sell new modules (Enterprise Plan)"
        elif row['churn_risk'] == "High":
            return "Assign Dedicated CSM & Offer 20% Discount"
        elif row['segment_label'] == "Dormant":
            return "Automated Win-back Email Sequence"
        else:
            return "Standard Quarterly Business Review"
            
    df['next_best_action'] = df.apply(action, axis=1)
    return df

def train_forecast(transactions):
    daily = transactions.groupby('date')['amount'].sum().reset_index()
    daily.columns = ['ds', 'y']
    
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(daily)
    
    future = model.make_future_dataframe(periods=90)
    forecast = model.predict(future)
    
    joblib.dump(model, 'models/prophet.joblib')
    forecast.to_csv('data/forecast_90d.csv', index=False)
    daily.to_csv('data/daily_revenue.csv', index=False)
    return forecast

if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_csv("data/customers_rfm.csv")
    tx = pd.read_csv("data/transactions.csv")
    
    print("Training Segmentation...")
    df = train_segmentation(df)
    
    print("Training Churn Prediction...")
    df = train_churn(df)
    
    print("Training CLV Model...")
    df = train_clv(df)
    
    print("Generating Recommendations...")
    df = generate_recommendations(df)
    
    df.to_csv("data/enterprise_intelligence.csv", index=False)
    
    print("Training Forecast...")
    train_forecast(tx)
    print("All backend pipelines complete.")

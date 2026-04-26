import pandas as pd
import numpy as np
import os
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from prophet import Prophet
import warnings

warnings.filterwarnings('ignore')

def train_segmentation_model(rfm_df, n_clusters=4):
    """Trains a KMeans clustering model on RFM data."""
    features = ['recency', 'frequency', 'monetary']
    X = rfm_df[features]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    # Assign labels
    rfm_df['cluster'] = kmeans.labels_
    
    # Save models
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/rfm_scaler.joblib')
    joblib.dump(kmeans, 'models/kmeans_model.joblib')
    
    # Provide human-readable labels based on cluster centers
    # Higher monetary & frequency, lower recency -> Best customers
    cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=features)
    cluster_centers['cluster'] = range(n_clusters)
    
    # We will score them to create labels
    cluster_centers['score'] = cluster_centers['frequency'] * cluster_centers['monetary'] / (cluster_centers['recency'] + 1)
    sorted_clusters = cluster_centers.sort_values(by='score', ascending=False)['cluster'].tolist()
    
    labels_map = {
        sorted_clusters[0]: "Champions",
        sorted_clusters[1]: "Loyal Customers",
        sorted_clusters[2]: "At Risk",
        sorted_clusters[3]: "Hibernating"
    }
    
    rfm_df['segment_label'] = rfm_df['cluster'].map(labels_map)
    rfm_df.to_csv("data/customers_segmented.csv", index=False)
    
    return rfm_df, kmeans

def train_churn_model(df):
    """Trains a RandomForest for Churn Prediction."""
    # Features and Target
    features = ['recency', 'frequency', 'monetary']
    X = df[features]
    y = df['churned']
    
    # In a real app we'd split train/test. For MVP, we train on all to get probability.
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    rf_model.fit(X, y)
    
    # Add predictions
    df['churn_prob'] = rf_model.predict_proba(X)[:, 1]
    
    def assign_risk(prob):
        if prob < 0.3:
            return "Low"
        elif prob < 0.7:
            return "Medium"
        else:
            return "High"
            
    df['churn_risk'] = df['churn_prob'].apply(assign_risk)
    
    # Save model
    joblib.dump(rf_model, 'models/churn_rf_model.joblib')
    df.to_csv("data/customers_churn_scored.csv", index=False)
    
    return df, rf_model

def train_forecast_model(transactions_df):
    """Trains a Prophet model for sales forecasting."""
    # Prepare data for Prophet
    daily_sales = transactions_df.groupby('date')['amount'].sum().reset_index()
    daily_sales.columns = ['ds', 'y']
    
    # Initialize and fit
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(daily_sales)
    
    # Make future dataframe (30 days)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    
    # Save
    joblib.dump(model, 'models/prophet_model.joblib')
    forecast.to_csv("data/sales_forecast.csv", index=False)
    
    return forecast, model

if __name__ == "__main__":
    print("Training ML Models...")
    
    # Ensure data exists
    if not os.path.exists('data/customers_rfm.csv'):
        print("Run data_gen.py first.")
        exit()
        
    df = pd.read_csv("data/customers_rfm.csv")
    transactions = pd.read_csv("data/transactions.csv")
    transactions['date'] = pd.to_datetime(transactions['date']).dt.date
    
    print("1. Training Segmentation Model...")
    train_segmentation_model(df.copy())
    
    print("2. Training Churn Prediction Model...")
    train_churn_model(df.copy())
    
    print("3. Training Forecast Model...")
    train_forecast_model(transactions)
    
    print("All models trained and saved successfully.")

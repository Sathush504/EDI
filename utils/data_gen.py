import pandas as pd
import numpy as np
import os
import urllib.request

def download_and_process_real_data():
    """Downloads and processes the UCI Online Retail Dataset."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
    filepath = "data/online_retail.xlsx"
    
    os.makedirs("data", exist_ok=True)
    
    if not os.path.exists(filepath):
        print(f"Downloading real dataset from {url}...")
        urllib.request.urlretrieve(url, filepath)
        print("Download complete.")
    else:
        print("Dataset already exists locally.")
        
    print("Loading and cleaning dataset (this may take a minute)...")
    df = pd.read_excel(filepath)
    
    # Clean data
    df = df.dropna(subset=['CustomerID']) # Remove missing customer IDs
    df = df[df['Quantity'] > 0] # Remove returns/canceled transactions
    
    # Calculate Revenue amount
    df['amount'] = df['Quantity'] * df['UnitPrice']
    
    # Map to our standard schema
    df = df.rename(columns={
        'CustomerID': 'customer_id',
        'InvoiceNo': 'transaction_id',
        'InvoiceDate': 'date',
        'Description': 'category'
    })
    
    # --- TIME SHIFT LOGIC ---
    # Shift all dates forward so the max date is today (April 2026)
    from datetime import datetime
    max_date = df['date'].max()
    time_diff = datetime.now() - max_date
    df['date'] = df['date'] + time_diff
    # ------------------------
    
    df['customer_id'] = df['customer_id'].astype(int).astype(str)
    
    # Create customer profiles
    customers_df = df[['customer_id', 'Country']].drop_duplicates(subset=['customer_id'])
    customers_df = customers_df.rename(columns={'Country': 'primary_category'}) # Treat country as category for our schema
    
    # Keep only necessary transaction columns
    transactions_df = df[['transaction_id', 'customer_id', 'date', 'amount', 'category']]
    
    return customers_df, transactions_df

def calculate_rfm_and_clv(customers_df, transactions_df):
    """Calculates RFM, Churn, and future value (CLV proxy)."""
    current_date = transactions_df['date'].max()
    
    rfm = transactions_df.groupby('customer_id').agg({
        'date': lambda x: (current_date - x.max()).days,
        'transaction_id': 'nunique', # Frequency: number of unique invoices
        'amount': 'sum'
    }).reset_index()
    
    rfm.rename(columns={'date': 'recency', 'transaction_id': 'frequency', 'amount': 'monetary'}, inplace=True)
    
    # Filter out outliers or negative monetary just in case
    rfm = rfm[rfm['monetary'] > 0]
    
    # Churn Label (Rule-based: > 180 days = Churned for this retail dataset)
    rfm['churned'] = (rfm['recency'] > 180).astype(int)
    
    # Add noise
    np.random.seed(42)
    noise_idx = np.random.choice(rfm.index, size=int(len(rfm)*0.05), replace=False)
    rfm.loc[noise_idx, 'churned'] = 1 - rfm.loc[noise_idx, 'churned']
    
    # Target CLV (Future 12 month revenue proxy, historical + growth noise)
    rfm['target_clv'] = rfm['monetary'] * np.random.uniform(0.8, 1.3, size=len(rfm))
    
    final_df = rfm.merge(customers_df, on='customer_id')
    return final_df

if __name__ == "__main__":
    customers, transactions = download_and_process_real_data()
    print("Calculating RFM & CLV...")
    final_df = calculate_rfm_and_clv(customers, transactions)
    
    final_df.to_csv("data/customers_rfm.csv", index=False)
    transactions.to_csv("data/transactions.csv", index=False)
    print(f"Processed {len(final_df)} unique customers and {len(transactions)} transactions.")
    print("Enterprise data (Real UCI Dataset) generated successfully.")

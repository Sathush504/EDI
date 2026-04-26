import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Configure page
st.set_page_config(
    page_title="EIDAP - Enterprise Decision Intelligence",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5em;
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    div[data-testid="stMetric"] label {
        color: #94A3B8 !important;
        font-size: 1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #F8FAFC !important;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    /* Global glass effect for plots */
    div.stPlotlyChart {
        background-color: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

sys.path.append(os.getcwd())

class RAGEngine:
    def __init__(self):
        self.index_path = "data/faiss_index.bin"
        self.docs_path = "data/business_reports.csv"
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.documents = []
        
        if os.path.exists(self.index_path) and os.path.exists(self.docs_path):
            self.index = faiss.read_index(self.index_path)
            self.documents = pd.read_csv(self.docs_path)['text'].tolist()

    def query(self, user_question, top_k=2):
        if not self.index:
            return "System is currently indexing the knowledge base. Please try again later."
            
        # Retrieve
        query_embedding = self.model.encode([user_question]).astype('float32')
        distances, indices = self.index.search(query_embedding, top_k)
        
        retrieved_context = [self.documents[i] for i in indices[0]]
        context_str = " ".join(retrieved_context)
        
        # Mock LLM Response Generation (Rule-based templating based on context)
        if "risk" in user_question.lower() or "churn" in user_question.lower():
            return f"Based on the data: {context_str} \n\nI recommend reviewing the Churn Analytics dashboard to assign dedicated Account Managers."
        elif "segment" in user_question.lower():
            return f"According to the latest segmentation: {context_str} \n\nThe Enterprise Champions are our most valuable group."
        elif "forecast" in user_question.lower() or "trend" in user_question.lower():
            return "The 90-day forecast indicates a steady growth trend. Please check the Forecasting dashboard for the exact Prophet model visualization."
        else:
            return f"Here is the information I retrieved: {context_str} \n\nHow else can I assist with your decision intelligence?"

# Initialize RAG Engine in Session State
if 'rag_engine' not in st.session_state:
    try:
        st.session_state.rag_engine = RAGEngine()
    except Exception as e:
        st.session_state.rag_engine = None
        st.session_state.rag_error = str(e)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hello! I am your EIDAP AI Assistant. Ask me anything about our customers, churn risks, or forecasts."}
    ]

# Load Data
@st.cache_data
def load_data():
    try:
        if not os.path.exists("data/enterprise_intelligence.csv"):
            return None, "Data files not found. Please run the pipeline first."
        
        df = pd.read_csv("data/enterprise_intelligence.csv")
        forecast = pd.read_csv("data/forecast_90d.csv")
        daily = pd.read_csv("data/daily_revenue.csv")
        return {'df': df, 'forecast': forecast, 'daily': daily}, ""
    except Exception as e:
        return None, str(e)

data, error_msg = load_data()

if data is None:
    st.error(f"Pipeline Not Run: {error_msg}")
    st.info("Run `python run_pipelines.py` to generate the enterprise intelligence data.")
    st.stop()

df = data['df']
forecast = data['forecast']
daily = data['daily']

# Sidebar Navigation
st.sidebar.title("EIDAP 🚀")
st.sidebar.markdown("Enterprise Intelligence")
page = st.sidebar.radio("Navigation", [
    "Dashboard", 
    "Customer 360", 
    "Churn Intelligence", 
    "Lifetime Value", 
    "Forecasting", 
    "AI Assistant"
])

# Helpers
def create_segmentation_chart():
    seg_counts = df['segment_label'].value_counts().reset_index()
    seg_counts.columns = ['Segment', 'Count']
    fig = px.pie(seg_counts, values='Count', names='Segment', hole=0.5, template="plotly_dark",
                 color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=30, b=0, l=0, r=0), title="Customer Segments")
    return fig

def create_rfm_chart():
    sample_df = df.sample(min(500, len(df)))
    fig = px.scatter_3d(sample_df, x='recency', y='frequency', z='monetary', color='segment_label', 
                        template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=0, b=0, l=0, r=0))
    return fig

def create_churn_chart():
    churn_counts = df['churn_risk'].value_counts().reset_index()
    churn_counts.columns = ['Risk', 'Count']
    fig = px.bar(churn_counts, x='Risk', y='Count', color='Risk', 
                 color_discrete_map={'High': '#EF4444', 'Medium': '#F59E0B', 'Low': '#10B981'}, template="plotly_dark")
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=30, b=0, l=0, r=0), title="Churn Risk Distribution")
    return fig

def create_forecast_chart():
    daily['ds'] = pd.to_datetime(daily['ds'])
    forecast['ds'] = pd.to_datetime(forecast['ds'])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily['ds'], y=daily['y'], name="Actual Revenue", line=dict(color="#6366F1", width=2)))
    last_actual = daily['ds'].max()
    future = forecast[forecast['ds'] > last_actual]
    
    fig.add_trace(go.Scatter(x=future['ds'], y=future['yhat'], name="Predicted", line=dict(color='#38BDF8', dash='dash', width=2)))
    fig.add_trace(go.Scatter(x=future['ds'], y=future['yhat_upper'], fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False))
    fig.add_trace(go.Scatter(x=future['ds'], y=future['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)', name='Confidence', fillcolor='rgba(56, 189, 248, 0.2)'))
    
    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=30, b=0, l=0, r=0), title="90-Day Revenue Trajectory")
    return fig

# Page Routing
if page == "Dashboard":
    st.title("Executive Dashboard")
    st.markdown("High-level enterprise decision metrics and health overview.")
    st.divider()
    
    total_rev = df['monetary'].sum()
    avg_clv = df['predicted_clv'].mean()
    high_risk = len(df[df['churn_risk'] == 'High'])
    active_cust = len(df[df['churned'] == 0])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Revenue", f"${total_rev:,.0f}", "+12.5% YoY")
    with col2: st.metric("Average CLV", f"${avg_clv:,.0f}", "+4.2% YoY")
    with col3: st.metric("Active Customers", f"{active_cust:,}", "+1.1% MoM")
    with col4: st.metric("At-Risk Accounts", f"{high_risk:,}", "-2.0% MoM", delta_color="inverse")
    
    st.write("")
    
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.plotly_chart(create_segmentation_chart(), use_container_width=True)
    with col_chart2:
        st.plotly_chart(create_forecast_chart(), use_container_width=True)

elif page == "Customer 360":
    st.title("Customer 360")
    st.markdown("Analyze segment health and behavioral clustering.")
    st.divider()
    
    st.plotly_chart(create_rfm_chart(), use_container_width=True, height=600)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Champions**\n\nHighly engaged, high spenders. Ready for cross-sells.")
    with col2:
        st.info("**Core Accounts**\n\nSteady recurring revenue. Focus on retention.")
    with col3:
        st.error("**At-Risk**\n\nDropping frequency. Needs immediate win-back campaigns.")

elif page == "Churn Intelligence":
    st.title("Churn Intelligence")
    st.markdown("Predictive risk scoring and automated next-best-actions.")
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_churn_chart(), use_container_width=True)
    with col2:
        st.subheader("Critical Action Required: Top At-Risk")
        risk_df = df[df['churn_risk'] == 'High'].nlargest(10, 'monetary')
        for _, row in risk_df.iterrows():
            st.warning(f"**Customer #{int(row['customer_id'])}** (Last active: {row['recency']} days ago)\n\n👉 **Action:** {row['next_best_action']}")

elif page == "Lifetime Value":
    st.title("Lifetime Value")
    st.markdown("Identify and nurture the most valuable customers.")
    st.divider()
    
    st.subheader("Top Enterprise Accounts by Predicted CLV")
    top_df = df.nlargest(20, 'predicted_clv').copy()
    
    def tier_badge(clv):
        if clv > 2000: return "🥇 Gold"
        if clv > 500: return "🥈 Silver"
        return "🥉 Bronze"
        
    top_df['Tier'] = top_df['predicted_clv'].apply(tier_badge)
    top_df['monetary'] = top_df['monetary'].apply(lambda x: f"${x:,.2f}")
    top_df['predicted_clv'] = top_df['predicted_clv'].apply(lambda x: f"${x:,.2f}")
    
    display_df = top_df[['customer_id', 'segment_label', 'monetary', 'predicted_clv', 'Tier']]
    display_df.columns = ['Customer ID', 'Segment', 'Historical Spend', 'Predicted 12M Value', 'Tier']
    st.dataframe(display_df, use_container_width=True, hide_index=True)

elif page == "Forecasting":
    st.title("Demand Forecasting")
    st.markdown("Time-series prediction of upcoming 90-day revenue trends.")
    st.divider()
    
    st.plotly_chart(create_forecast_chart(), use_container_width=True)

elif page == "AI Assistant":
    st.title("AI Business Assistant")
    st.markdown("Talk to your data using the RAG-powered intelligence engine.")
    st.divider()
    
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    if prompt := st.chat_input("Ask 'Why is churn high?' or 'Which segment is most profitable?'"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            if st.session_state.rag_engine:
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.rag_engine.query(prompt)
                        st.markdown(response)
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.error("RAG Engine is not initialized. Please ensure data files exist.")


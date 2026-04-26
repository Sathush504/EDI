import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

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
            
    def build_knowledge_base(self):
        print("Building Knowledge Base for RAG...")
        # Synthesize business reports from data
        try:
            df = pd.read_csv("data/enterprise_intelligence.csv")
            
            reports = []
            
            # Overview report
            total_rev = df['monetary'].sum()
            avg_clv = df['predicted_clv'].mean()
            reports.append(f"The total historical revenue is ${total_rev:,.2f}. The average predicted Customer Lifetime Value (CLV) is ${avg_clv:,.2f}.")
            
            # Segment report
            for seg in df['segment_label'].unique():
                count = len(df[df['segment_label'] == seg])
                reports.append(f"There are {count} customers in the {seg} segment.")
                
            # Churn report
            high_risk = len(df[df['churn_risk'] == 'High'])
            reports.append(f"There are currently {high_risk} customers at High risk of churning.")
            
            # Action report
            for action in df['next_best_action'].unique():
                count = len(df[df['next_best_action'] == action])
                reports.append(f"The recommended action '{action}' applies to {count} customers.")
                
            self.documents = reports
            
            # Create FAISS Index
            embeddings = self.model.encode(self.documents)
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings.astype('float32'))
            
            # Save
            faiss.write_index(self.index, self.index_path)
            pd.DataFrame({'text': self.documents}).to_csv(self.docs_path, index=False)
            print("Knowledge base built successfully.")
            
        except FileNotFoundError:
            print("Run ml_pipelines.py first to generate intelligence data.")

    def query(self, user_question, top_k=2):
        if not self.index:
            return "System is currently indexing the knowledge base. Please try again later."
            
        # Retrieve
        query_embedding = self.model.encode([user_question]).astype('float32')
        distances, indices = self.index.search(query_embedding, top_k)
        
        retrieved_context = [self.documents[i] for i in indices[0]]
        context_str = " ".join(retrieved_context)
        
        # Mock LLM Response Generation (Rule-based templating based on context)
        # In production, we would pass `context_str` to OpenAI/Anthropic.
        
        if "risk" in user_question.lower() or "churn" in user_question.lower():
            return f"Based on the data: {context_str} \n\nI recommend reviewing the Churn Analytics dashboard to assign dedicated Account Managers."
        elif "segment" in user_question.lower():
            return f"According to the latest segmentation: {context_str} \n\nThe Enterprise Champions are our most valuable group."
        elif "forecast" in user_question.lower() or "trend" in user_question.lower():
            return "The 90-day forecast indicates a steady growth trend. Please check the Forecasting dashboard for the exact Prophet model visualization."
        else:
            return f"Here is the information I retrieved: {context_str} \n\nHow else can I assist with your decision intelligence?"

if __name__ == "__main__":
    engine = RAGEngine()
    engine.build_knowledge_base()

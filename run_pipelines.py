import os
import subprocess

def run_script(script_name):
    print(f"Running {script_name}...")
    result = subprocess.run(['python', script_name], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {script_name}: {result.stderr}")
        exit(1)
    else:
        print(result.stdout)

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    print("=== STEP 1: Generate Enterprise Data ===")
    run_script('utils/data_gen.py')
    
    print("=== STEP 2: Train ML Models & Pipelines ===")
    run_script('backend/ml_pipelines.py')
    
    print("=== STEP 3: Build RAG FAISS Index ===")
    run_script('chatbot/rag_engine.py')
    
    print("All backend processes completed successfully! Ready for Reflex Frontend.")

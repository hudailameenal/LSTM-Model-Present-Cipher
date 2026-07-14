import os
import csv
from tabulate import tabulate
from tensorflow.keras.models import load_model

def display_saved_results():
    results_path = os.path.join("results", "rwtt_round_results.csv")
    
    if not os.path.exists(results_path):
        print("No results file found!")
        return

    table_data = []
    with open(results_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            table_data.append([
                row['Dataset'],
                row['Round'],
                row['Optimizer'],
                row['Epochs'],
                row['Lr_rate'],
                row['Hidden_Nodes'],
                row['Bitwise_test_acc'] + "%",
                row['Mean_byte_error']
            ])

    headers = [
        "Dataset", "Round", "Optimizer", "Epochs", 
        "Learning Rate", "Hidden Nodes", "Accuracy", "MBE"
    ]
    
    print("\n=== RWTT Training Results ===\n")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

def display_model_info():
    """Display information about the saved RWTT models"""
    datasets = ['dataset5', 'dataset6', 'dataset7']
    
    for dataset in datasets:
        model_path = os.path.join("models", f"rwtt_{dataset}_final.h5")
        
        if not os.path.exists(model_path):
            print(f"\n⚠ Model file not found at {model_path}")
            continue
        
        print(f"\n=== RWTT Model Information ({dataset}) ===")
        try:
            model = load_model(model_path)
            print("✓ Model loaded successfully")
            
            model.summary()
            
            print(f"\nOptimizer: {model.optimizer.__class__.__name__}")
            print(f"Loss function: {model.loss}")
            
            if hasattr(model, 'metrics_names'):
                print(f"Metrics: {model.metrics_names}")
                
        except Exception as e:
            print(f"Error loading model: {e}")

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    display_saved_results()

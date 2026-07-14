import os
import csv
import numpy as np
from tabulate import tabulate
from tensorflow.keras.models import load_model

def display_saved_results():
    results_path = os.path.join("results", "nctt_round_results.csv")
    
    if not os.path.exists(results_path):
        print("No results file found!")
        return

    # Prepare table headers
    headers = [
        "Round", "Optimizer", "Epochs", "Learning Rate", "Hidden Nodes",
        "Dataset2 Acc", "Dataset2 MBE",
        "Dataset3 Acc", "Dataset3 MBE",
        "Dataset4 Acc", "Dataset4 MBE"
    ]
    table = []

    with open(results_path, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            table.append([
                i,
                row.get("Optimizer", ""),
                row.get("Epochs", ""),
                row.get("Lr_rate", ""),
                row.get("Hidden_Nodes", ""),
                row.get("Test_Accuracy_dataset2", ""),
                row.get("Mean_Byte_Error_dataset2", ""),
                row.get("Test_Accuracy_dataset3", ""),
                row.get("Mean_Byte_Error_dataset3", ""),
                row.get("Test_Accuracy_dataset4", ""),
                row.get("Mean_Byte_Error_dataset4", "")
            ])

    print("\n=== Saved NCTT Model Results ===")
    print(tabulate(table, headers=headers, tablefmt="grid"))

def display_model_info():
    """Display information about the saved NCTT model"""
    model_path = os.path.join("models", "present_nctt.h5")
    
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        return
    
    print("\n=== NCTT Model Information ===")
    
    try:
        model = load_model(model_path)
        print("✓ Model loaded successfully\n")
    
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

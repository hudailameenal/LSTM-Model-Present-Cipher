import os
import csv
import numpy as np
from tensorflow.keras.models import load_model

def display_saved_results():
    """Display the saved results from the CSV file in tabular form"""
    results_file = os.path.join("results", "ctt_round_results.csv")
    
    # Check if results file exists
    if not os.path.exists(results_file):
        print(f"Results file not found at {results_file}")
        return
    
    # Load results into a list
    rows = []
    with open(results_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    if not rows:
        print("No results found in CSV file.")
        return
    
    # Print header
    headers = ["Round", "Optimizer", "Epochs", "Lr_rate", "Hidden_Nodes", "Test_Accuracy", "Mean_Byte_Error"]
    print("=== Saved Model Results ===")
    print("-" * 90)
    print("{:<8} {:<10} {:<8} {:<10} {:<14} {:<14} {:<14}".format(*headers))
    print("-" * 90)
    
    # Print each row
    for row in rows:
        print("{:<8} {:<10} {:<8} {:<10} {:<14} {:<14} {:<14}".format(
            row['Round'],
            row['Optimizer'],
            row['Epochs'],
            row['Lr_rate'],
            row['Hidden_Nodes'],
            row['Test_Accuracy'],
            row['Mean_Byte_Error']
        ))
    print("-" * 90)

def display_model_info():
    """Display information about the saved model"""
    model_path = os.path.join("models", "present_ctt.h5")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        return
    
    print("\n=== Model Information ===")
    
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

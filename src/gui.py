
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import numpy as np
from model_training import load_data, generate_embeddings, apply_smote, train_and_evaluate_model, save_evaluation_results, save_model_and_embeddings

# Function to run the model from the GUI
def run_model():
    try:
        # Load the dataset
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if not file_path:
            return
        
        data = load_data(file_path)
        
        # Prepare data
        X = data['Claim Description']
        y = data['Coverage Code']  # You can modify this to 'Accident Source' if needed
        
        # Generate embeddings
        X_embeddings = generate_embeddings(X)
        
        # Apply SMOTE
        X_resampled, y_resampled = apply_smote(X_embeddings, y)
        
        # Compute class weights
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight('balanced', classes=np.unique(y_resampled), y=y_resampled)
        class_weight_dict = dict(zip(np.unique(y_resampled), class_weights))
        
        # Split data for training and evaluation
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)
        
        # Train and evaluate the model
        report, roc_auc, precision, recall, model = train_and_evaluate_model(X_train, y_train, X_test, y_test, class_weight_dict)
        
        # Save the model and embeddings
        model_path = "output/model/optimized_random_forest_model.pkl"
        embeddings_path = "output/embeddings/X_train_embeddings.pkl"
        save_model_and_embeddings(model, X_embeddings, model_path, embeddings_path)
        
        # Display evaluation results
        result_text = f"Precision: {precision:.4f}\nRecall: {recall:.4f}\nROC AUC: {roc_auc:.4f}"
        result_label.config(text=result_text)
        
        # Save results to Excel
        output_folder = "output/reports/evaluation_results.xlsx"
        save_evaluation_results(report, roc_auc, precision, recall, output_folder)
        
        messagebox.showinfo("Success", f"Model executed successfully. Results saved to {output_folder}")
    
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Setting up the GUI window
root = tk.Tk()
root.title("Insurance Claims Model")

# GUI Layout
frame = tk.Frame(root)
frame.pack(padx=20, pady=20)

label = tk.Label(frame, text="Select Dataset and Run Model", font=("Arial", 16))
label.grid(row=0, column=0, pady=10)

run_button = tk.Button(frame, text="Run", command=run_model, font=("Arial", 14))
run_button.grid(row=1, column=0, pady=10)

result_label = tk.Label(frame, text="Evaluation Results will appear here", font=("Arial", 12))
result_label.grid(row=2, column=0, pady=10)

# Main loop to run the application
root.mainloop()

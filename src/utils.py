

import os
import pandas as pd
import numpy as np

# Function to create directories if they don't exist
def create_directories(directories):
    """Create directories if they don't exist."""
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
    print(f"Directories created: {directories}")

# Function to load data with error handling
def load_data_with_error_handling(file_path, file_type="xlsx"):
    """Load data from Excel or CSV with error handling."""
    try:
        if file_type == "xlsx":
            data = pd.read_excel(file_path)
        elif file_type == "csv":
            data = pd.read_csv(file_path)
        else:
            raise ValueError("Unsupported file type")
        print(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

# Function to save any dataframe as Excel with multiple sheets
def save_dataframe_to_excel(dataframes, file_path):
    """Save multiple dataframes to an Excel file."""
    try:
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            for sheet_name, df in dataframes.items():
                df.to_excel(writer, sheet_name=sheet_name)
        print(f"Data saved to {file_path}")
    except Exception as e:
        print(f"Error saving data: {e}")

# Function to handle missing data (optional)
def handle_missing_data(df):
    """Handle missing values in the dataset."""
    # For demonstration, fill missing values with the mode of each column
    return df.fillna(df.mode().iloc[0])

# Function to save model with error handling
def save_model_with_error_handling(model, file_path):
    """Save a model with error handling."""
    try:
        import pickle
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved successfully at {file_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

# Function to load model with error handling
def load_model_with_error_handling(file_path):
    """Load a model from a file with error handling."""
    try:
        import pickle
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded successfully from {file_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

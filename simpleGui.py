import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import pickle
from xgboost import XGBClassifier
import datetime
import os

# For graphs stuff
from graphs import disaster_frequency_by_region, extent_of_disasters_by_region, choropleth_damage_and_deaths, plot_roc_curve


# Load the ensemble model and scaler
with open("ensemble_model.pkl", "rb") as file:
    ensemble_model = pickle.load(file)
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Load test data for ROC curve
with open("xgb_model.pkl", "rb") as file:
    xgb_model = pickle.load(file)  
with open("X_test_scaled.pkl", "rb") as file:
    X_test_scaled = pickle.load(file)
with open("y_test.pkl", "rb") as file:
    y_test = pickle.load(file)

# Load the processed data and original features
df_processed = pd.read_csv('processedNaturalDisasters.csv')
original_features = list(df_processed.select_dtypes(exclude=['object']).drop(columns=['Disaster Occurred']).columns)

# Extract unique countries from the processed data
countries_in_data = [col.replace("Country_", "") for col in df_processed.columns if col.startswith("Country_")]
unique_countries = sorted(countries_in_data)  # Ensure it's sorted

# Mapping of month names to numbers
month_mapping = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}

# Function to predict all disasters
def predict_all_disasters(ensemble_model, scaler, unique_countries, start_year, end_year):
    predictions = []
    for country in unique_countries:
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                input_dict = {
                    "Year": year,
                    "Month": month,
                    "Duration": 0  
                }
                input_df = pd.DataFrame([input_dict])

                # Create one-hot encoded country columns
                country_columns = {f"Country_{c}": 0 for c in unique_countries}  # Default all to 0
                if f"Country_{country}" in country_columns:
                    country_columns[f"Country_{country}"] = 1  # Set selected country to 1

                country_df = pd.DataFrame([country_columns])
                input_df = pd.concat([input_df, country_df], axis=1)

                for col in original_features:
                    if col not in input_df.columns:
                        input_df[col] = 0

                input_df = input_df[original_features]  # Keep column order
                input_scaled = scaler.transform(input_df)
                pred = ensemble_model.predict(input_scaled)[0]

                if pred == 1:
                    predictions.append((year, month, country))

    predictions.sort(key=lambda x: (x[0], x[1]))  # Sort by year, then month
    return predictions

# Function to find the next and disaster predictions
def find_next_n_predictions(ensemble_model, scaler, unique_countries, n, start_year, start_month):
        predictions = []
        year = start_year
        month = start_month
        # Continue until we have n predictions
        while len(predictions) < n:
            for country in unique_countries:
                input_dict = {
                    "Year": year,
                    "Month": month,
                    "Duration": 0  
                }
                input_df = pd.DataFrame([input_dict])
                
                # Create one-hot encoded country columns
                country_columns = {f"Country_{c}": 0 for c in unique_countries}
                
                # Ensure the selected country exists in the columns
                if f"Country_{country}" in country_columns: 
                    country_columns[f"Country_{country}"] = 1
                country_df = pd.DataFrame([country_columns])
                input_df = pd.concat([input_df, country_df], axis=1)

                # Ensure all original features are present
                for col in original_features:
                    if col not in input_df.columns:
                        input_df[col] = 0
                input_df = input_df[original_features]

                # Scale the input data
                input_scaled = scaler.transform(input_df)
                pred = ensemble_model.predict(input_scaled)[0]
                
                # If disaster is predicted, add to the list
                if pred == 1:
                    predictions.append((year, month, country))
                    if len(predictions) >= n:
                        break
            # Increment if needed
            month += 1
            if month > 12:
                month = 1
                year += 1
        return predictions

# Function to open a file
def open_file(file_path):
    if os.path.exists(file_path):
        os.startfile(file_path)
    else:
        messagebox.showerror("File Not Found", f"File '{file_path}' does not exist.")

# Define the GUI class
class DisasterPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Disaster Prediction")
        self.root.geometry("500x700")

        ttk.Label(root, text="Disaster Occurrence Prediction", font=("Arial", 14)).pack(pady=10)

        # Country Selection
        ttk.Label(root, text="Select Country:").pack()
        self.country_var = tk.StringVar()
        self.country_dropdown = ttk.Combobox(root, textvariable=self.country_var, values=list(unique_countries), state="readonly")
        self.country_dropdown.pack(pady=5)

        # Month Selection
        ttk.Label(root, text="Select Month:").pack()
        self.month_var = tk.StringVar()
        self.month_dropdown = ttk.Combobox(root, textvariable=self.month_var, values=list(month_mapping.keys()), state="readonly")
        self.month_dropdown.pack(pady=5)

        # Year Selection
        ttk.Label(root, text="Select Year:").pack()
        self.year_var = tk.StringVar()
        self.year_dropdown = ttk.Combobox(root, textvariable=self.year_var, values=[str(y) for y in range(2025, 2031)], state="readonly")
        self.year_dropdown.pack(pady=5)

        self.predict_button = ttk.Button(root, text="Predict", command=self.make_prediction)
        self.predict_button.pack(pady=10)

        # New button to show the next 10 predicted disasters
        self.next10_button = ttk.Button(root, text="Show Next 10 Disasters", command=self.show_next_10_disasters)
        self.next10_button.pack(pady=5)

        self.result_label = ttk.Label(root, text="Predictions:", font=("Arial", 12, "bold"))
        self.result_label.pack(pady=5)

        # Display multiple predictions
        self.result_listbox = tk.Listbox(root, height=10, width=50)
        self.result_listbox.pack(pady=10)

        self.view_data_label = ttk.Label(root, text="View historical data graphs", font=("Arial", 12, "bold"))
        self.view_data_label.pack(pady=5)

        # Button to display the disaster frequency graph
        self.graph_button = ttk.Button(root, text="Show Disaster Frequency Graph", command=disaster_frequency_by_region)
        self.graph_button.pack(pady=5)

        # Button to display extent of disaster by region graph
        self.extent_graph_button = ttk.Button(root, text="Show Extent of Disasters Graph", command=extent_of_disasters_by_region)
        self.extent_graph_button.pack(pady=5)

        # Button to show map
        self.map_button = ttk.Button(root, text="Show Damage Map", command=choropleth_damage_and_deaths)
        self.map_button.pack(pady=5)

        # Button for showing ROC curve
        self.roc_curve_button = ttk.Button(root, text="Show ROC Curve (Ensemble)", command=lambda: plot_roc_curve(ensemble_model, X_test_scaled, y_test))
        self.roc_curve_button.pack(pady=5)

        # Button for viewing confusion matrices
        self.view_confusion_button = ttk.Button(root, text="View Confusion Matrices", command=lambda: open_file("confusion_matrices.png"))
        self.view_confusion_button.pack(pady=5)

        # Button for viewing training vs test accuracy
        self.view_accuracy_button = ttk.Button(root, text="View Train vs Test Accuracy", command=lambda: open_file("train_vs_test_accuracy.png"))
        self.view_accuracy_button.pack(pady=5)
        
    # Function to make a prediction
    def make_prediction(self):
        selected_country = self.country_var.get().strip()
        selected_month = self.month_var.get()
        selected_year = self.year_var.get()

        if not selected_country or not selected_month or not selected_year:
            messagebox.showwarning("Input Error", "Please select a country, month, and year.")
            return

        input_dict = {
            "Year": int(selected_year),
            "Month": month_mapping[selected_month],
            "Duration": 0
        }
        input_df = pd.DataFrame([input_dict])

        # Create one-hot encoded columns for all countries (ensuring they exist)
        country_columns = {f"Country_{c}": 0 for c in unique_countries} 
        if f"Country_{selected_country}" in country_columns:
            country_columns[f"Country_{selected_country}"] = 1 

        country_df = pd.DataFrame([country_columns])
        input_df = pd.concat([input_df, country_df], axis=1)

        for col in original_features:
            if col not in input_df.columns:
                input_df[col] = 0

        # Keep the original column order
        input_df = input_df[original_features]  
        input_scaled = scaler.transform(input_df)
        pred = ensemble_model.predict(input_scaled)[0]

        # Display the prediction
        self.result_listbox.delete(0, tk.END)  # Clear previous results
        if pred == 1:
            self.result_listbox.insert(tk.END, f"Prediction: Disaster in {selected_country} ({selected_month} {selected_year})")
        else:
            self.result_listbox.insert(tk.END, "No disaster predicted for this selection.")
    
    # Function to show the next 10 disasters
    def show_next_10_disasters(self):
        # Get current year and month
        now = datetime.datetime.now()
        current_year = now.year
        current_month = now.month

        # Find the next 10 disaster predictions from the current date
        next10 = find_next_n_predictions(ensemble_model, scaler, unique_countries, 10, current_year, current_month)

        self.result_listbox.delete(0, tk.END)  # Clear previous results
        if next10:
            for year, month, country in next10:
                self.result_listbox.insert(tk.END, f"{country} - {month}/{year}")
        else:
            self.result_listbox.insert(tk.END, "No future disasters predicted in the selected range.")


# Main function
if __name__ == "__main__":
    root = tk.Tk()  # Create the main window
    app = DisasterPredictionApp(root) # Create the application
    root.mainloop() # Start the event loop

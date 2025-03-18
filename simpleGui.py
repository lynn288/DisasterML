import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import pickle
from xgboost import XGBClassifier
import datetime

# ===================================
# 1. Load the Trained XGBoost Model and Scaler
# ===================================
with open("xgb_model.pkl", "rb") as file:
    xgb_model = pickle.load(file)
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Load the processed CSV to obtain the original feature names
df_processed = pd.read_csv('processedNaturalDisasters.csv')
original_features = list(df_processed.select_dtypes(exclude=['object']).drop(columns=['Disaster Occurred']).columns)

# Get unique countries from the processed dataset (to match training)
countries_in_data = [col.replace("Country_", "") for col in df_processed.columns if col.startswith("Country_")]
unique_countries = sorted(countries_in_data)  # Ensure it's sorted

# Month mapping for dropdown selection
month_mapping = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}

# ===================================
# 2. Predict All Future Disasters
# ===================================
def predict_all_disasters(xgb_model, scaler, unique_countries, start_year, end_year):
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

                # Ensure all features match the trained model
                for col in original_features:
                    if col not in input_df.columns:
                        input_df[col] = 0

                input_df = input_df[original_features]  # Keep column order
                input_scaled = scaler.transform(input_df)
                pred = xgb_model.predict(input_scaled)[0]

                if pred == 1:
                    predictions.append((year, month, country))

    predictions.sort(key=lambda x: (x[0], x[1]))  # Sort by year, then month
    return predictions

# Get predictions for the next 5 years
current_year = datetime.datetime.now().year
future_end_year = current_year + 5  
future_disasters = predict_all_disasters(xgb_model, scaler, unique_countries, current_year, future_end_year)

# Print predicted disasters in the console
print("\n### All Future Predicted Disasters ###")
if future_disasters:
    for year, month, country in future_disasters:
        print(f"- {country}: {month}/{year}")
else:
    print("No disasters predicted in the next 5 years.")

# ===================================
# 3. Define Tkinter GUI
# ===================================
class DisasterPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Disaster Prediction")
        self.root.geometry("500x400")

        ttk.Label(root, text="Disaster Occurrence Prediction", font=("Arial", 14)).pack(pady=10)

        # Country Selection
        ttk.Label(root, text="Select Country:").pack()
        self.country_var = tk.StringVar()
        self.country_dropdown = ttk.Combobox(root, textvariable=self.country_var, values=list(unique_countries))
        self.country_dropdown.pack(pady=5)

        # Month Selection
        ttk.Label(root, text="Select Month:").pack()
        self.month_var = tk.StringVar()
        self.month_dropdown = ttk.Combobox(root, textvariable=self.month_var, values=list(month_mapping.keys()))
        self.month_dropdown.pack(pady=5)

        # Year Selection
        ttk.Label(root, text="Select Year:").pack()
        self.year_var = tk.StringVar()
        self.year_dropdown = ttk.Combobox(root, textvariable=self.year_var, values=[str(y) for y in range(2024, 2031)])
        self.year_dropdown.pack(pady=5)

        self.predict_button = ttk.Button(root, text="Predict", command=self.make_prediction)
        self.predict_button.pack(pady=10)

        self.result_label = ttk.Label(root, text="Predictions:", font=("Arial", 12, "bold"))
        self.result_label.pack(pady=5)

        # Listbox to display multiple predictions
        self.result_listbox = tk.Listbox(root, height=10, width=50)
        self.result_listbox.pack(pady=10)

    def make_prediction(self):
        selected_country = self.country_var.get().strip()
        selected_month = self.month_var.get()
        selected_year = self.year_var.get()

        if not selected_country or not selected_month or not selected_year:
            messagebox.showwarning("Input Error", "Please select a country, month, and year.")
            return

        # Create the input dataframe
        input_dict = {
            "Year": int(selected_year),
            "Month": month_mapping[selected_month],
            "Duration": 0
        }
        input_df = pd.DataFrame([input_dict])

        # Create one-hot encoded columns for all countries (ensuring they exist)
        country_columns = {f"Country_{c}": 0 for c in unique_countries}  # Default all to 0
        if f"Country_{selected_country}" in country_columns:
            country_columns[f"Country_{selected_country}"] = 1  # Set selected country to 1

        country_df = pd.DataFrame([country_columns])
        input_df = pd.concat([input_df, country_df], axis=1)

        # Ensure all features match the trained model
        for col in original_features:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[original_features]  # Keep column order
        input_scaled = scaler.transform(input_df)
        pred = xgb_model.predict(input_scaled)[0]

        self.result_listbox.delete(0, tk.END)  # Clear previous results
        if pred == 1:
            self.result_listbox.insert(tk.END, f"Prediction: Disaster in {selected_country} ({selected_month} {selected_year})")
        else:
            self.result_listbox.insert(tk.END, "No disaster predicted for this selection.")

# ===================================
# 4. Run the GUI
# ===================================
if __name__ == "__main__":
    root = tk.Tk()
    app = DisasterPredictionApp(root)
    root.mainloop()

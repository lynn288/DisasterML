import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import pickle
from xgboost import XGBClassifier

# ===================================
# 1. Load the Trained XGBoost Model
# ===================================
with open("xgboost_model.pkl", "rb") as file:
    xgb_model = pickle.load(file)

# Load dataset to extract necessary details
df = pd.read_excel('naturalDisasters.xlsx')

# Extract unique country names (ensure uniformity)
df["Country"] = df["Country"].str.strip()  # Remove spaces
unique_countries = df["Country"].dropna().unique()
unique_countries.sort()

# Extract unique years from dataset
unique_years = df["Start Year"].dropna().astype(int).unique()
unique_years.sort()

# Define month mapping
month_mapping = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}

# Get the original training features for one-hot encoding consistency
original_features = list(pd.get_dummies(df.drop(columns=["Disaster Type"]), drop_first=True).columns)

# ===================================
# 2. Define Tkinter GUI
# ===================================
class DisasterPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Natural Disaster Prediction")
        self.root.geometry("450x350")

        # Title Label
        ttk.Label(root, text="Disaster Prediction", font=("Arial", 14)).pack(pady=10)

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
        self.year_dropdown = ttk.Combobox(root, textvariable=self.year_var, values=[str(year) for year in range(2024, 2035)])  # Future years
        self.year_dropdown.pack(pady=5)

        # Predict Button
        self.predict_button = ttk.Button(root, text="Predict", command=self.make_prediction)
        self.predict_button.pack(pady=10)

        # Result Label
        self.result_label = ttk.Label(root, text="", font=("Arial", 12, "bold"))
        self.result_label.pack(pady=10)

    # =========================
    # 3. Prediction Function
    # =========================
    def make_prediction(self):
        # Get user inputs
        selected_country = self.country_var.get().strip()
        selected_month = self.month_var.get()
        selected_year = self.year_var.get()

        # Validate inputs
        if not selected_country or not selected_month or not selected_year:
            messagebox.showwarning("Input Error", "Please select a country, month, and year.")
            return

        # Ensure country exists in the dataset
        if selected_country not in unique_countries:
            messagebox.showerror("Error", f"Country '{selected_country}' not found in dataset.")
            return

        # Convert categorical inputs into numerical values
        month_encoded = month_mapping[selected_month]  # Convert month name to number
        year_encoded = int(selected_year)  # Convert year to integer

        # Create DataFrame for input features
        input_data = pd.DataFrame(columns=original_features)  # Empty DataFrame with training feature names

        # Fill numeric values
        input_data.loc[0, "Start Year"] = year_encoded
        input_data.loc[0, "Start Month"] = month_encoded

        # Handle country one-hot encoding (set 1 for selected country, 0 for others)
        country_column = f"Country_{selected_country}"
        if country_column in input_data.columns:
            input_data.loc[0, country_column] = 1

        # Fill missing one-hot encoded columns with 0
        input_data.fillna(0, inplace=True)

        # Ensure correct feature order
        input_data = input_data[original_features]

        # Convert to NumPy array for prediction
        input_array = input_data.to_numpy()

        # Make prediction
        predicted_class = xgb_model.predict(input_array)[0]

        # Convert numeric prediction back to disaster type
        disaster_mapping = {
            0: "No Disaster", 1: "Drought", 2: "Earthquake", 3: "Epidemic", 
            4: "Extreme Temperature", 5: "Flood", 6: "Glacial Lake Outburst Flood", 
            7: "Impact", 8: "Infestation", 9: "Mass Movement (Dry)", 
            10: "Mass Movement (Wet)", 11: "Storm", 12: "Volcanic Activity", 13: "Wildfire"
        }

        predicted_disaster = disaster_mapping.get(predicted_class, "Unknown Disaster")

        # Display Result
        self.result_label.config(text=f"Prediction: {predicted_disaster}", foreground="blue")


# ===================================
# 4. Run the GUI
# ===================================
if __name__ == "__main__":
    root = tk.Tk()
    app = DisasterPredictionApp(root)
    root.mainloop()

import subprocess
import threading
import os
import time

# Declaration of paths
BASE_DIR = "."  # Change this to the respective directory

DATA_PROCESSING_SCRIPT = os.path.join(BASE_DIR, "dataProcessing.py")  # Data processing script
MODEL_TRAINING_SCRIPT = os.path.join(BASE_DIR, "modelTrainer.py")  # Model training script
GUI_SCRIPT = os.path.join(BASE_DIR, "simpleGui.py")  # GUI script

DATA_FILE = os.path.join(BASE_DIR, "naturalDisasters.xlsx")  # Dataset file
PROCESSED_DATA_FILE = os.path.join(BASE_DIR, "processedNaturalDisasters.csv")  # Processed dataset file

# Model files
MODEL_FILES = {
    "XGBoost Model": os.path.join(BASE_DIR, "xgb_model.pkl"), # XGBoost model
    "CatBoost Model": os.path.join(BASE_DIR, "catboost_model.pkl"), # CatBoost model
    "Random Forest Model": os.path.join(BASE_DIR, "rf_model.pkl"), # Random Forest model
    "Scaler": os.path.join(BASE_DIR, "scaler.pkl") # Scaler
}


# Function to run a Python script using subprocess
def run_script(script_path):
    if os.path.exists(script_path):
        print(f"\nRunning {script_path} ...")
        start_time = time.time()
        result = subprocess.run(["python", script_path], check=True)
        elapsed_time = time.time() - start_time
        print(f"\n{script_path} completed in {elapsed_time:.2f} seconds.")
        return result
    else:
        print(f"ERROR: Script not found: {script_path}")


# Main function
if __name__ == "__main__":

    # Step 1: Check Data Files & Run Data Processing if Needed
    if not os.path.exists(PROCESSED_DATA_FILE):
        print("\nStarting data preprocessing...")
        processing_thread = threading.Thread(target=run_script, args=(DATA_PROCESSING_SCRIPT,))
        processing_thread.start()
        processing_thread.join()  # Ensure data processing completes before moving to model training
    else:
        print(f"Processed data file already exists: {PROCESSED_DATA_FILE}, skipping preprocessing.")

    # Step 2: Check If Model Files exists & Run Model Training if Needed
    missing_models = [name for name, path in MODEL_FILES.items() if not os.path.exists(path)]

    if missing_models:
        print("\nThe following model files are missing:") # Model files are missing
        for model in missing_models:
            print(f"- {model}") # Print missing model files
        print("\nStarting model training...")

        training_thread = threading.Thread(target=run_script, args=(MODEL_TRAINING_SCRIPT,))
        training_thread.start()
        training_thread.join()  # Ensure model training completes before launching GUI
    else:
        print("All model files are present. Skipping model training.")

    # Step 3: Run GUI
    print("\nLaunching Disaster Prediction GUI...")
    gui_thread = threading.Thread(target=run_script, args=(GUI_SCRIPT,))
    gui_thread.start()

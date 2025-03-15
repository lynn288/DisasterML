import subprocess
import threading
import os
import time

# ==========================================
# 1. Define File Paths
# ==========================================
BASE_DIR = "INF2008-main" #Change this to the respective directory

DATA_PROCESSING_SCRIPT = os.path.join(BASE_DIR, "dataProcessing.py")
MODEL_TRAINING_SCRIPT = os.path.join(BASE_DIR, "naturalDisastersV2.py")
GUI_SCRIPT = os.path.join(BASE_DIR, "simpleGui.py")

DATA_FILE = os.path.join(BASE_DIR, "naturalDisasters.xlsx") #Make sure these files are also in the BASE_DIR
PROCESSED_DATA_FILE = os.path.join(BASE_DIR, "processedNaturalDisasters.csv")
XGB_MODEL_FILE = os.path.join(BASE_DIR, "xgb_model.pkl")


# ==========================================
# 2. Utility Function to Run Python Scripts
# ==========================================
def run_script(script_path):
    """
    Executes a given Python script using subprocess.
    
    Args:
        script_path (str): Full path of the Python script to be executed.
    """
    if os.path.exists(script_path):
        print(f"\nRunning {script_path} ...")
        start_time = time.time()
        result = subprocess.run(["python", script_path], check=True)
        elapsed_time = time.time() - start_time
        print(f"\n{script_path} completed in {elapsed_time:.2f} seconds.")
        return result
    else:
        print(f"ERROR Script not found: {script_path}")


# ==========================================
# 3. Main Execution Flow
# ==========================================
if __name__ == "__main__":
    
    # ========================
    # Step 1: Run Data Processing
    # ========================
    if not os.path.exists(PROCESSED_DATA_FILE):
        print("\nStarting data preprocessing...")
        processing_thread = threading.Thread(target=run_script, args=(DATA_PROCESSING_SCRIPT,))
        processing_thread.start()
        processing_thread.join()  # Ensure data processing completes before moving to model training
    else:
        print(f"Processed data file already exists: {PROCESSED_DATA_FILE}, skipping preprocessing.")

    # ========================
    # Step 2: Run Model Training
    # ========================
    if not os.path.exists(XGB_MODEL_FILE):
        print("\nStarting model training...")
        training_thread = threading.Thread(target=run_script, args=(MODEL_TRAINING_SCRIPT,))
        training_thread.start()
        training_thread.join()  # Ensure model training completes before launching GUI
    else:
        print(f"Trained model already exists: {XGB_MODEL_FILE}, skipping model training.")

    # ========================
    # Step 3: Run GUI
    # ========================
    print("\nLaunching Disaster Prediction GUI...")
    gui_thread = threading.Thread(target=run_script, args=(GUI_SCRIPT,))
    gui_thread.start()

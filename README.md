# Disaster Prediction and Analysis Project

---

## Table of Contents

- [Requirements](#requirements)
- [Installation / Setup](#installation)
- [Usage](#usage)
  - [Running the Project](#running-the-project)
  - [Using the GUI](#using-the-gui)
- [Troubleshooting](#troubleshooting)

## Requirements

This project is runs on Python 3.12.X (What was used by us)

The following Python packages are required (Provided in the requirements.txt file)

- pandas>=1.3.0
- numpy>=1.21.0
- scikit-learn>=0.24.0
- matplotlib>=3.3.0
- seaborn>=0.11.0
- xgboost>=1.4.0
- catboost>=0.26.0
- imbalanced-learn>=0.8.0

---

## Installation / Setup

### 1. Clone the repo:
```bash
git clone https://github.com/yourusername/your-repository.git
cd your-repository
```

### 2. Install dependencies:
```bash
pip install -r requirementx.txt
```

### 3. Ensure that "naturalDisasters.xlsx" is inside the same directory.


## Usage
The whole project can be started from the "main.py" file.

**1. Data Processing:**
If "processedNaturalDisasters.csv" does not exist, "dataProcessing.py" is executed to process the raw data.

**2. Model Training:**
If the trained model file does not exist, "naturalDisastersV2.py" is executed to train the model.

**3. Launching the GUI:**
Finally, "simpleGui.py" is launched.

### To run the project:
```bash
python main.py
```

## Using the GUI
Once the GUI launches, you can

- **Make a Prediction:**
Select a country, month, and year and click the "Predict" button. The result will be displayed below.

- **View Future Disaster Predictions:**
Click the "Show Next 10 Disasters" button to list upcoming predicted disaster events.

- **View graphs and additional information:**
    Using the following buttons:
    - **Show Disaster Frequency Graph:**
      Displays a bar chart of disaster frequency by region with a time slider.
    - **Show Extent of Disasters Graph:** Displays a dual-axis bar chart (total deaths and total damage in trillions USD) by region with a slider.
    - **Show Damage Map:** Opens a choropleth map of disaster damage by country.
    - **Show ROC Curve (XGB):** Plots the ROC curve for the XGBoost model.
    - **View Confusion Matrices:** Opens the confusion matrices image.
    - **View Train vs Test Accuracy:** Opens the accuracy comparison chart.

## Troubleshooting

### File Not Found Errors:
- Ensure that all required files (e.g., naturalDisasters.xlsx) are in the same directory.

### Model Version Compatibility:
- Make sure you are using Python 3.12.X, as CatBoost requires Python versions below 3.13.X.

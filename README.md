# Disaster Prediction and Analysis Project

---

## Table of Contents

- [Requirements](#requirements)
- [Installation / Setup](#installation/Setup)
- [Usage](#usage)
  - [Running the Project](#runningTheproject)
  - [Using the GUI](#gui)
- [Project Structure](#project-structure)
- [Notes](#notes)
- [Troubleshooting](#troubleshooting)
- [Authors](#authors)
- [Appendix: EM-DAT Dataset Documentation](#appendix-em-dat-dataset-documentation)


## Description
This is a collaborative project created by Singapore Institute of Technology (SIT) students to fulfill module requirements. The project's purpose is to create and implement a machine learning system that forecasts natural disasters using historical data. Our workflow covers the entire cycle, from problem formulation and data collection to data processing, feature engineering, model training, performance evaluation, and results analysis. To increase prediction accuracy, we use a combination of machine learning methods (including Random Forest, XGBoost, and CatBoost) and an ensemble voting classifier, as well as tackling issues like class imbalance. A user-friendly graphical interface has been created to aid disaster prediction and data visualization, making the system more accessible and practical for real-world applications.

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

<a name="installation/setup"></a>
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
If the trained model files does not exist, "modelTrainer.py" is executed to train the model.

**3. Launching the GUI:**
Finally, "simpleGui.py" is launched.

<a name="runningTheproject"></a>
### To run the project:
```bash
python main.py
```
<a name="gui"></a>
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

## Project Structure
dataProcessing.py – Loads raw data, processes it, and saves the aggregated CSV file.

modelTrainer.py – Loads processed data, splits it by date, trains multiple ML models, evaluates performance, and saves models.

simpleGui.py – Provides a user interface for making predictions and visualizing disaster data.

graphs.py – Contains functions for generating various graphs (disaster frequency, extent, damage maps, ROC curves).

### Model Evaluation
The project includes a robust model evaluation pipeline to ensure the reliability of predictions. The evaluation process considers multiple metrics:

- **Accuracy**: Measures overall correctness of predictions.
- **Precision & Recall**: Precision helps reduce false positives, while recall ensures actual disaster events are captured.
- **F1 Score**: Balances precision and recall for a comprehensive measure of model performance.
- **Confusion Matrices**: Provides a breakdown of true positives, false positives, false negatives, and true negatives for each model.

The models evaluated include:
- **Random Forest**
- **XGBoost**
- **CatBoost**
- **Ensemble Model (Voting Classifier)**

The **Ensemble Model** was selected as the final model due to its **balanced tradeoff between precision, recall, and generalization ability**, making it the most reliable choice for disaster prediction. While it has slightly higher computational requirements, the performance benefits outweigh the minimal overhead.

Graphs and confusion matrices can be accessed in the GUI under:
- **View Confusion Matrices**
- **View Train vs Test Accuracy**

## Notes
Date-based Splitting:
The model training uses a date-based split to ensure that training data only comes from past records. This mimics real-world forecasting and avoids data leakage.

Handling Imbalanced Data:
The scripts use undersampling and SMOTE to balance the data, as the dataset has many more instances with no disaster than with a disaster.

Model Ensemble:
An ensemble Voting Classifier is built to combine predictions from Random Forest, XGBoost, and CatBoost for improved performance.

## Troubleshooting

### File Not Found Errors:
- Ensure that all required files (e.g., naturalDisasters.xlsx) are in the same directory.

### Model Version Compatibility:
- Make sure you are using Python 3.12.X, as CatBoost requires Python versions below 3.13.X.

## Authors

- [@Joshyua](https://github.com/Joshyua) - Joshua Ho
- [@lynn288](https://github.com/lynn288) - Jocelyn Yeo
- [@imbored1313](https://github.com/imbored1313) - Bryce Tan
- [@2103300Git](https://github.com/2103300Git) - Bryan Toh
- [@ZLYX1](https://github.com/ZLYX1) - Zola Lim
- [@xnicro](https://github.com/xnicro) - Muhammad Solikhin

## Appendix: EM-DAT Dataset Documentation

1. Introduction
The Emergency Events Database (EM-DAT) is a comprehensive database maintained by the Centre for Research on the Epidemiology of Disasters (CRED). It contains detailed records of natural and technological disasters worldwide. This dataset has been used in our project to analyze historical disaster events and develop a machine learning model for predicting future disaster occurrences.

2. Data Source and Licensing
Source:
The EM-DAT dataset is maintained by the Centre for Research on the Epidemiology of Disasters (CRED).
Website: http://www.emdat.be/

Licensing:
The dataset is provided for research purposes. Users must adhere to CRED’s data usage policies and give appropriate credit in any publications or presentations.

3. Data Collection and Updates
Collection Methods:
Data in EM-DAT is collected from a wide range of sources including UN agencies, government institutions, and non-governmental organizations (NGOs).
Updates:
The database is updated regularly to incorporate new disaster events and revisions to historical data. For our project, we used a version extracted on [insert extraction date/version here].

4. Variables and Definitions
Key variables used from the EM-DAT dataset in our project include:

Disaster Type:
Classification of the disaster (e.g., earthquake, flood, storm).

Country:
The country where the disaster occurred.

Start Year, Start Month, Start Day:
Components that represent the start date of the disaster event.

End Year, End Month, End Day:
Components that represent the end date of the disaster event.

Disaster Group and Disaster Subgroup:
Additional categorizations that provide context and further classification of disaster types.


5. Data Preprocessing
The raw EM-DAT data underwent several preprocessing steps to ensure it was suitable for model training:

Cleaning and Missing Value Treatment:
Missing values in date components were replaced with placeholder values (e.g., -1) before conversion.

Date Conversion:
The date components were combined to create proper date objects for analysis.

Feature Engineering:
A new feature, Duration, was calculated as the difference (in days) between the disaster’s end date and start date.

Aggregation:
Data was aggregated on a monthly basis per country to enable time-series analysis and forecasting.

Encoding:
Categorical variables, such as Country, Disaster Group, and Disaster Subgroup, were one-hot encoded or label encoded as appropriate.

Feature Selection:
Features with a low correlation to the target variable (disaster occurrence) were dropped, while ensuring that essential date-related features (Year and Month) were retained.
The processed dataset is saved as processedNaturalDisasters.csv and serves as the basis for model training and evaluation in our project.

6. Limitations and Considerations
Data Completeness:
Some regions may experience underreporting or delays in reporting disaster events, which could affect data quality.

Temporal Changes:
Variations in data collection methods and reporting standards over time may influence trends in the dataset.

Interpretation:
While EM-DAT is a reputable source, it is important to consider external factors (e.g., socio-economic conditions, climate change) that may not be fully captured in the dataset but could influence disaster occurrence.

7. References
Centre for Research on the Epidemiology of Disasters (CRED). EM-DAT: The International Disaster Database. Retrieved from http://www.emdat.be/
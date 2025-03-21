import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import plotly.express as px
from matplotlib.widgets import Slider
from sklearn.metrics import roc_curve, auc

# Function to plot disaster frequency by region
def disaster_frequency_by_region():
    # Load dataset
    df = pd.read_excel("naturalDisasters.xlsx")

    # Ensure "Start Year" is in data
    if "Start Year" not in df.columns:
        print("No 'Start Year' column found. Please verify your dataset.")
        return

    # Group by Region & Start Year, count disasters
    region_year_counts = df.groupby(["Region", "Start Year"]).size().reset_index(name="Disaster Count")

    # Build a list of valid years, exclude 2024
    all_years = sorted(region_year_counts["Start Year"].unique())
    years = [y for y in all_years if y < 2024]  # Filter out 2024 (or >= 2024 if needed)

    if not years:
        print("No valid years found in the dataset.")
        return

    # figure & axis for the bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.3)

    # Plot function for a given year
    def plot_year_data(year):
        ax.clear()
        data_for_year = region_year_counts[region_year_counts["Start Year"] == year]
        ax.bar(data_for_year["Region"], data_for_year["Disaster Count"], color='skyblue')
        ax.set_xlabel("Region")
        ax.set_ylabel("Number of Disasters")
        ax.set_title(f"Disaster Frequency by Region in {year}")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Initial plot (first year in the filtered list)
    initial_year = years[0]
    plot_year_data(initial_year)

    # slider axis
    slider_ax = plt.axes([0.1, 0.08, 0.8, 0.05], facecolor='lightgoldenrodyellow')

    # Create slider
    year_slider = Slider(
        ax=slider_ax,
        label='',            
        valmin=years[0],
        valmax=years[-1],
        valinit=initial_year,
        valstep=1,
        valfmt=''            
    )

    year_slider.valtext.set_visible(False)

    # Add tick labels (the list of years) along the slider
    slider_ax.set_xticks(years)
    slider_ax.set_xticklabels([str(y) for y in years], rotation=45, ha='center')

    fig.text(0.5, 0.02, "Move the slider to change the year", ha='center', va='center') 
    
    # Define update function
    def update(val):
        current_year = int(year_slider.val)
        plot_year_data(current_year)
        fig.canvas.draw_idle()

    year_slider.on_changed(update)

    plt.show()



# Function to plot extent of disasters by region
def extent_of_disasters_by_region():
    # Load dataset
    df = pd.read_excel("naturalDisasters.xlsx")

    # Ensure there's a "Start Year" column
    if "Start Year" not in df.columns:
        print("No 'Start Year' column found. Please verify your dataset.")
        return

    # exclude 2024
    df = df[df["Start Year"] < 2024]

    # 4. Group by Region & Start Year, sum total deaths and damage
    grouped = df.groupby(["Region", "Start Year"]).agg({
        "Total Deaths": "sum",
        "Total Damage, Adjusted ('000 US$)": "sum"
    }).reset_index()

    # convert cost from '000 US$ to trillions of USD
    grouped["Damage_Trillions"] = grouped["Total Damage, Adjusted ('000 US$)"] / 1e9

    # sorted list of unique years
    years = sorted(grouped["Start Year"].unique())
    if not years:
        print("No valid years found in the dataset.")
        return

    # Create figure
    fig, ax1 = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(bottom=0.3)

    # Twin axis for the right y-axis
    ax2 = ax1.twinx()

    def plot_year_data(year):
        ax1.clear()
        ax2.clear()

        data_for_year = grouped[grouped["Start Year"] == year]

        # Create positions for bars
        regions = data_for_year["Region"]
        x = np.arange(len(regions))
        width = 0.4

        # Left axis = total deaths 
        ax1.bar(x - width/2, data_for_year["Total Deaths"], width, label="Total Deaths", color="red")
        ax1.set_ylabel("Total Deaths", color="red")

        # Right axis = total damage 
        ax2.bar(x + width/2, data_for_year["Damage_Trillions"], width, label="Total Damage (Adjusted)", color="blue")
        ax2.set_ylabel("Total Damage (Trillions USD)", color="blue", labelpad=20)
        # label is on right side with enough space
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()

        # X-axis region labels
        ax1.set_xlabel("Region")
        ax1.set_xticks(x)
        ax1.set_xticklabels(regions, rotation=45)

        def to_trillions_label(val, pos):
            return f"{val:,.2f}T"
        ax2.yaxis.set_major_formatter(ticker.FuncFormatter(to_trillions_label))

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        ax1.set_title(f"Extent of Disasters by Region in {year}")

    # Plot the initial year
    initial_year = years[0]
    plot_year_data(initial_year)

    # Create slider axis
    slider_ax = plt.axes([0.15, 0.1, 0.7, 0.05], facecolor='lightgoldenrodyellow')

    # Create slider
    year_slider = Slider(
        ax=slider_ax,
        label='',            
        valmin=years[0],
        valmax=years[-1],
        valinit=initial_year,
        valstep=1,
        valfmt=''         
    )

    year_slider.valtext.set_visible(False)

    # Add years along the slider axis
    slider_ax.set_xticks(years)
    slider_ax.set_xticklabels([str(y) for y in years], rotation=45, ha='center')

    def update(val):
        current_year = int(year_slider.val)
        plot_year_data(current_year)
        fig.canvas.draw_idle()

    year_slider.on_changed(update)

    plt.show()

# Function to create a choropleth map of total disaster damage and deaths by country over time
def choropleth_damage_and_deaths():
    df = pd.read_excel("naturalDisasters.xlsx")
    
    if "Start Year" not in df.columns:
        print("No 'Start Year' column found. Please verify your dataset.")
        return
    
    df = df[df["Start Year"] <= 2023]

    
    # Group by ISO, Country, and Start Year, sum damage and deaths
    grouped = df.groupby(["ISO", "Country", "Start Year"]).agg({
        "Total Damage, Adjusted ('000 US$)": "sum",
        "Total Deaths": "sum"
    }).reset_index()
    
    # Convert damage from thousands of USD to billions of USD
    grouped["Damage_Billions"] = grouped["Total Damage, Adjusted ('000 US$)"] / 1e6
    
    fig = px.choropleth(
        grouped,
        locations="ISO",                  
        color="Damage_Billions",          
        hover_name="Country",             
        hover_data={
            "Damage_Billions": ":.2f",    
            "Total Deaths": True,        
            "Start Year": True          
        },
        color_continuous_scale="Reds",
        labels={
            "Damage_Billions": "Total Damage (Billions USD)",
            "Total Deaths": "Total Deaths",
            "Start Year": "Year"
        },
        title="Total Disaster Damage & Deaths by Country Over Time",
        animation_frame="Start Year"      
    )
    
    fig.update_layout(
        coloraxis_colorbar=dict(
            title="Damage (Billions USD)"
        )
    )
    
    fig.show()

# Function to plot ROC curve for a given model
def plot_roc_curve(model, X_test, y_test):

    # Convert y_test to a binary integer array
    y_test = np.array(y_test).ravel().astype(int)

    # Get predicted probabilities for positive class
    y_scores = model.predict_proba(X_test)[:, 1]
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for Ensemble Model')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()



# Main function
if __name__ == '__main__':
    disaster_frequency_by_region()
    extent_of_disasters_by_region()
    choropleth_damage_and_deaths()
    plot_roc_curve()

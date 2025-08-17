import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =========================================================================
# === CONFIGURATION =======================================================
# =========================================================================

# --- Input File Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_LOSS_CSV_PATH = os.path.join(BASE_DIR, "..", "runs", "CAH_run_csv", "train_loss.csv")
VAL_LOSS_CSV_PATH = os.path.join(BASE_DIR, "..", "runs", "CAH_run_csv", "validation_loss.csv")

# --- Output File Path ---
OUTPUT_PLOT_PATH = os.path.join(BASE_DIR, "..","data", "illustrative_images", "plots", "CAH_model_training_loss.png")

# --- Plotting Settings ---
PLOT_TITLE = 'CAH Model Training and Validation Loss Over Epochs'
X_AXIS_LABEL = 'Epoch'
Y_AXIS_LABEL = 'Mean Squared Error (MSE) Loss'
# =========================================================================


def create_loss_plot():
    """
    Loads training and validation loss data from TensorBoard CSVs
    and creates a publication-quality plot.
    """
    print("Loading data...")
    try:
        # The CSVs have three columns: 'Wall time', 'Step', and 'Value'.
        train_df = pd.read_csv(TRAIN_LOSS_CSV_PATH)
        val_df = pd.read_csv(VAL_LOSS_CSV_PATH)
    except FileNotFoundError as e:
        print(f"FATAL ERROR: Could not find data files. Make sure they are in the project root.")
        print(f"Missing file: {e.filename}")
        return

    print("Creating plot...")
    
    # Use seaborn for a professional plot style.
    sns.set_theme(style="whitegrid")
    
    # Create a figure and axes object. This gives us more control.
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot the training and validation loss on the same axes
    ax.plot(train_df['Step'], train_df['Value'], label='Training Loss', color='blue', linewidth=2)
    ax.plot(val_df['Step'], val_df['Value'], label='Validation Loss', color='orange', linewidth=2)
    
    # --- Add professional labels, title, etc. ---
    ax.set_title(PLOT_TITLE, fontsize=16, weight='bold')
    ax.set_xlabel(X_AXIS_LABEL, fontsize=12)
    ax.set_ylabel(Y_AXIS_LABEL, fontsize=12)
    ax.legend(fontsize=12)
    
    # You can set the y-axis to a logarithmic scale if the initial loss is very high
    # ax.set_yscale('log')
    
    # Set limits to zoom in on the interesting part of the graph if needed
    # For example, to ignore the initial high loss from the first few epochs
    # ax.set_ylim(0, 70) 
    
    # Ensure the layout is tight to prevent labels from being cut off
    plt.tight_layout()
    
    # Save the figure with high resolution, suitable for a thesis
    plt.savefig(OUTPUT_PLOT_PATH, dpi=300)
    
    print(f"\nPlot saved successfully to: {OUTPUT_PLOT_PATH}")

    # Print the best epoch
    best_epoch_step = val_df.loc[val_df['Value'].idxmin()]['Step']
    print("Epoch with the lowest validation loss: " + str(best_epoch_step))
    
    # Display the plot in a pop-up window for immediate viewing
    plt.show()

if __name__ == "__main__":
    create_loss_plot()
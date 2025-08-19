import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =========================================================================
# === CONFIGURATION =======================================================
# =========================================================================

# --- Input ---
train_mse = [0.02563400499833127, 0.00788576900862002, 0.0031651268950857532, 0.002243166506620279, 0.0018374250241322443, 0.0016419830814508412, 0.0009861094384784033, 0.0010004125724663027, 0.0015650459320265024, 0.0008007897222948184, 0.0008807224673849608, 0.002453046667263455, 0.0018693372194926875, 0.0007985555314614127, 0.0010380378150633381, 0.0012948642935953103, 0.0009295639470413637, 0.0012215695955092088, 0.0013770506954945934, 0.0030419626920775043]
val_mse = [0.0022232462807248035, 0.006694570843440791, 0.001506786301615648, 0.010957560346772274, 0.002591226932903131, 0.010385485660905639, 0.0012402721195636937, 0.00313537833862938, 0.0005968306378539031, 0.0021635377604980023, 0.003827856304512049, 0.0008457123801539031, 0.0008132779854349792, 0.0010192105847333247, 0.000598215750263383, 0.0016907806391827762, 0.0022277939637812476, 0.011105922933590287, 0.016983293771627359, 0.011330290356030066]
epochs = list(range(1, len(train_mse) + 1))

# --- Output File Path ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PLOT_PATH = os.path.join(BASE_DIR, "..","data", "illustrative_images", "plots", "CAH_training_loss_curve.png")

# --- Plotting Settings ---
PLOT_TITLE = 'Training and Validation Loss Over Epochs'
X_AXIS_LABEL = 'Epoch'
Y_AXIS_LABEL = 'Mean Squared Error (MSE) Loss'
# =========================================================================


def create_loss_plot():
    """
    Loads training and validation loss data from TensorBoard CSVs
    and creates a publication-quality plot.
    """
    print("Creating plot...")
    
    sns.set_theme(style="whitegrid")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(epochs, train_mse, label='Training Loss', color='blue', linewidth=2)
    ax.plot(epochs, val_mse, label='Validation Loss', color='orange', linewidth=2)

    ax.set_title(PLOT_TITLE, fontsize=16, weight='bold')
    ax.set_xlabel(X_AXIS_LABEL, fontsize=12)
    ax.set_ylabel(Y_AXIS_LABEL, fontsize=12)
    ax.legend(fontsize=12)
    
    # ax.set_yscale('log')
    
    # Ensure the layout is tight to prevent labels from being cut off
    plt.tight_layout()
    
    # Save the figure with high resolution, suitable for a thesis
    plt.savefig(OUTPUT_PLOT_PATH, dpi=300)
    
    print(f"\nPlot saved successfully to: {OUTPUT_PLOT_PATH}")

    # Print the best epoch
    # best_epoch_step = val_df.loc[val_df['Value'].idxmin()]['Step']
    # print("Epoch with the lowest validation loss: " + str(best_epoch_step))
    
    # Display the plot in a pop-up window for immediate viewing
    plt.show()

if __name__ == "__main__":
    create_loss_plot()
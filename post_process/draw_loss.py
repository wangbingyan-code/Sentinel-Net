import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def normalize_series(series):
    return (series - series.min()) / (series.max() - series.min())

def load_and_prepare_data(file_path, model_name):
    try:
        df = pd.read_csv(file_path)
        if 'epoch' not in df.columns or 'loss' not in df.columns:
            print(f"Warning: '{model_name}' data file is missing required columns.")
            return None
        df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')
        df['loss'] = pd.to_numeric(df['loss'], errors='coerce')
        df.dropna(subset=['epoch', 'loss'], inplace=True)
        df.sort_values('epoch', inplace=True)
        
        # Normalize loss
        df['loss'] = normalize_series(df['loss'])
        
        return df
    except Exception as e:
        print(f"Error loading '{model_name}': {e}")
        return None

AE_model = load_and_prepare_data('../output/loss/loss_data_ae.csv', 'AE_model')
AVE_model = load_and_prepare_data('../output/loss/loss_data_vae.csv', 'AVE_model')
Sentinel_net = load_and_prepare_data('../output/loss/loss_data_sentinel.csv', 'Sentinel_net')
AAE_model = load_and_prepare_data('../output/loss/loss_data_aae.csv', 'AAE_model')

# Check if there is valid data
models = [(AE_model, 'AE_model', 'b', '-', 2), 
          (AVE_model, 'AVE_model', 'g', '--', 2), 
          (AAE_model, 'AAE_model', 'purple', ':', 2), 
          (Sentinel_net, 'Sentinel_net', 'r', '-.', 2)]
models = [m for m in models if m[0] is not None]

if not models:
    print("No valid data to plot. Exiting.")
else:
    # Create a dual-axis plot
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot model loss curves
    for model, label, color, linestyle, linewidth in models[:-1]:
        ax1.plot(model['epoch'], model['loss'], label=f'{label}: Loss', color=color, linestyle=linestyle, linewidth=linewidth)

    ax1.set_xlabel('Epochs', fontsize=16)
    ax1.set_ylabel('Loss (AE, AVE, AAE)', fontsize=16, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True)

    ax1.xaxis.set_major_locator(MultipleLocator(10))  # Major tick every 10 epochs
    ax1.xaxis.set_minor_locator(MultipleLocator(5))   # Minor tick every 5 epochs

    # Plot Sentinel_net loss (right y-axis)
    if models[-1][1] == 'Sentinel_net':
        sentinel_data = models[-1][0]
        ax2 = ax1.twinx()
        ax2.plot(sentinel_data['epoch'], sentinel_data['loss'], label='Sentinel_net: Loss', color='r', linestyle='-.', linewidth=2)
        ax2.set_ylabel('Loss (Sentinel_net)', fontsize=16, color='red')
        ax2.tick_params(axis='y', labelcolor='red')

    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    if 'ax2' in locals():
        lines2, labels2 = ax2.get_legend_handles_labels()
        fig.legend(lines1 + lines2, labels1 + labels2, loc="upper right", bbox_to_anchor=(0.9, 0.9), fontsize=14)
    else:
        fig.legend(lines1, labels1, loc="upper right", bbox_to_anchor=(0.9, 0.9), fontsize=14)

    # Optimize layout
    plt.title('Loss Curves with Dual Axes', fontsize=18)
    plt.tight_layout()

    # Show
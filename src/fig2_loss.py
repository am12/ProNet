import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data preparation
data = {
    "Epoch": list(range(1, 17)),
    "Training Loss": [0.323, 0.129, 0.104, 0.119, 0.024, 0.009, 0.026, 0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.027, 0.018, 0.003],
    "Testing Loss": [0.261, 0.206, 0.111, 0.237, 0.186, 0.130, 0.083, 0.126, 0.113, 0.117, 0.136, 0.080, 0.168, 0.246, 0.241, 0.191]
}
df = pd.DataFrame(data)

# Melt the data to long format for seaborn
df_long = df.melt('Epoch', var_name='Type', value_name='Loss')

sns.set_palette('bright', 2)
# Plotting with seaborn
plt.figure(figsize=(10, 5))
sns.lineplot(data=df_long, x='Epoch', y='Loss', hue='Type', marker='o')
plt.title('Training and Testing Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

# Save the plot
plt.savefig('../results/fig/2_loss_plot_sns.png', dpi=300, bbox_inches='tight')
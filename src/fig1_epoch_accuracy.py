import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define the data
# data = {
#     'Epoch': [
#         'Epoch 1', 'Epoch 2', 'Epoch 3', 'Epoch 4', 'Epoch 5',
#         'Epoch 6', 'Epoch 7', 'Epoch 8', 'Epoch 9', 'Epoch 10',
#         'Epoch 11', 'Epoch 12', 'Epoch 13', 'Epoch 14', 'Epoch 15', 'Epoch 16'
#     ],
#     'Test Accuracy': [
#         0.8963666666666666, 0.921, 0.9273666666666667, 0.9489666666666666, 0.9556,
#         0.9663666666666667, 0.9749, 0.9765666666666667, 0.9782333333333333, 0.9782666666666666,
#         0.9783333333333334, 0.9702666666666667, 0.9608333333333333, 0.9633333333333334,
#         0.9335666666666667, 0.9528666666666666
#     ],
#     'Test Precision (Macro)': [
#         0.7849561785678588, 0.805685815615374, 0.7830558663106114, 0.8644859116987031, 0.824513229472454,
#         0.8828929921176123, 0.8912143758609654, 0.8938496583077479, 0.8998062669429122, 0.9018099429612704,
#         0.8985404431386467, 0.865749951379169, 0.8828422868426602, 0.8782091774212327,
#         0.8501389248568766, 0.8623359497287713
#     ],
#     'Test Recall (Macro)': [
#         0.7347264941782061, 0.7682082507829449, 0.7379050192683243, 0.8254993093590507, 0.7238854955333914,
#         0.8192709343831277, 0.8273501378995425, 0.8169802904209134, 0.8275193831189241, 0.8254759559780672,
#         0.8261054349542952, 0.7986100567468447, 0.79061834508196, 0.7882974733278242,
#         0.8058896015412895, 0.7671646911471374
#     ]
# }

# Convert the dictionary to a DataFrame
# df = pd.DataFrame(data)

df = pd.read_csv('../results/3/data.tsv', sep='\t')
print(df)

# Set the index to be the epochs for better visualization in the heatmap
df_long = df.melt(id_vars='Model', var_name='Metric', value_name='Value')

plt.figure(figsize=(14, 8))
sns.barplot(x='Model', y='Value', hue='Metric', data=df_long, palette='viridis')

plt.title('Model Metrics Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.ylim(0.5, 1)
plt.xticks(rotation=90)  # Rotate x labels for better visibility
plt.legend(title='Metric', title_fontsize='13', fontsize='11', loc='upper right')


# Save the plot
plt.savefig('../results/fig/1_epoch_accuracy_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
import pandas as pd
import matplotlib.pyplot as plt

# Load the data from a tab-separated .txt file
df = pd.read_csv('scores/element_analysis.txt', sep='\t')

# Create the plot
plt.figure(figsize=(8, 6))
plt.scatter(df['count_in_training_data'], df['average_score'], color='blue')

# Annotate each point with the element name
for _, row in df.iterrows():
    plt.text(row['count_in_training_data'], row['average_score'],
             row['element'], fontsize=10, ha='right', va='bottom')

# Label axes
plt.xlabel('Occurrence in training data', fontsize=20)
plt.ylabel('Accuracy in phase name inference', fontsize=20)

# Set x and y axis size
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# Set limits for x and y axes
plt.xlim(0, 50)
plt.ylim(0, 1)

# Show plot
plt.tight_layout()
plt.savefig('element_accuracy_vs_occurrence.png')

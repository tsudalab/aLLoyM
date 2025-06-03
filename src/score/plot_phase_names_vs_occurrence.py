import pandas as pd
import matplotlib.pyplot as plt

# example data
"""
element	average_score	count_in_training_data
Calcium	0.21880746	3741
Scandium	0.244069015	17885
Sodium	0.258503401	4684
Tantalum	0.384022645	26793
Copper	0.471878263	38117
Indium	0.508174387	7359
Chromium	0.52557368	31450
Vanadium	0.527942498	33870
Manganese	0.531346806	28693
Platinum	0.553427145	16289
Nickel	0.600928074	39340
Gadolinium	0.613691194	11431
Aluminium	0.616428963	42787
Thorium	0.636185243	0
Cerium	0.641756549	19214
Silver	0.658960944	26719
Niobium	0.660693642	22488
Gallium	0.665120594	26969
Zirconium	0.670008594	40445
Lanthanum	0.681829358	31751
Cobalt	0.69561398	41863
Iron	0.722222222	47559
Praseodymium	0.723602484	13806
Bismuth	0.726612371	15679
Tungsten	0.729356478	21253
Zinc	0.737566138	42409
Titanium	0.745569446	56605
Ytterbium	0.75907781	14433
Silicon	0.765840906	25640
Lead	0.766526442	8991
Antimony	0.766583403	26592
Boron	0.779985678	27284
Tin	0.794371222	23394
Gold	0.823996034	18503
Yttrium	0.824442944	19180
Molybdenum	0.834117075	37357
Dysprosium	0.839311164	17239
Palladium	0.83984041	28254
Thulium	0.853820598	7684
Neodymium	0.887836491	18829
Terbium	0.899556344	0
Lutetium	0.905759162	5950
Erbium	0.908418367	12500
Holmium	0.961961367	10740
"""

# Load the data from a tab-separated .txt file
df = pd.read_csv('/home/yoikawa/src/phase_LLM/aLLoyM/dataset/multi/split_random/mistral_LLM/tmp/phase_name_element_stats.csv', sep='\t')

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
plt.xlim(0, 60000)
plt.ylim(0, 1)

# Show plot
plt.tight_layout()
plt.savefig('element_accuracy_vs_occurrence.png')

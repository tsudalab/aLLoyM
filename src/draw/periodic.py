import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.pyplot import cm
from matplotlib.collections import PatchCollection

import numpy as np

from pymatgen.core.periodic_table import Element
from pymatgen.util.plotting import periodic_table_heatmap


import pathlib

elements = []

p_temp = pathlib.Path('dataset/CPDDB_data').glob('*.dat')

num = 0

for p in p_temp:
    file_name = p.name

    bina = file_name.split('_')

    for el in bina[0].split('-'):
        elements.append(el)

    num += 1
    
print("number of data =", num)
print(len(elements))

import collections

c = collections.Counter(elements)

print(c)


#random_data = {'Te': 30, 'Au': 10,
#                       'Th': 1, 'Ni': 30}

plt = periodic_table_heatmap(c, cmap="Blues", value_format=".0f", value_fontsize=14)
#plt = periodic_table_heatmap(random_data)

plt.show()
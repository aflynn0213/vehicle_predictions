# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 13:42:18 2023

@author: aflyn
"""

import pandas as pd

data = {'cola': [1, 2, 3, 4],
        'colb': [5, 6, 7, 8]}

df = pd.DataFrame(data)
df.index = ["W","X","R","P"]
# Create 'x' as a reference to the 'cola' column in the DataFrame
x = df.iloc[:,0]


# Modify the value in 'x'
x[0] = 45454
indices = df.index
print(df)
print(indices)
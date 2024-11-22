# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 00:51:27 2024

@author: geneh
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os

os.chdir('C:/Users/geneh/Desktop/ml/plot')

# Load both JSON files
with open('losses.json', 'r') as f1:
   losses1 = json.load(f1)
with open('losses6.json', 'r') as f2:
   losses2 = json.load(f2)

# Calculate average validation losses for each model
val_losses1 = np.mean([list(map(float, values)) for values in losses1['val_losses'].values()], axis=0)
val_losses2 = np.mean([list(map(float, values)) for values in losses2['val_losses'].values()], axis=0)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(val_losses1, label='4 Layers')
plt.plot(val_losses2, label='6 Layers')
plt.xlabel('Epoch')
plt.ylabel('RMSE Loss')
plt.title('Average Validation Loss Comparison')
plt.legend()
plt.grid(True)
plt.savefig('validation_loss_comparison.png')
plt.show()
plt.close()
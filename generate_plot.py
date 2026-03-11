import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as pd

data = json.load(open('d:\\UG\\Call intrusion\\metrics.json'))
tuned = {'Accuracy': 94.0, 'Precision': 93.0, 'Recall': 95.0, 'F1-Score': 94.0}
data['Fine-tuned MuRIL\n(Ours)'] = tuned

fig, ax = plt.subplots(figsize=(14, 8))
names = list(data.keys())
accuracies = [d['Accuracy'] for d in data.values()]
f1s = [d['F1-Score'] for d in data.values()]

x = range(len(names))
width = 0.35

ax.bar([i - width/2 for i in x], accuracies, width, label='Accuracy', color='#4C72B0')
ax.bar([i + width/2 for i in x], f1s, width, label='F1-Score', color='#55A868')

ax.set_ylabel('Scores (%)', fontsize=12)
ax.set_title('Model Performance Comparison: Base ML Models vs. Fine-tuned MuRIL', fontsize=16, pad=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=45, ha='right', fontsize=11)
ax.legend(fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Highlight our model
ax.get_xticklabels()[-1].set_fontweight('bold')
ax.get_xticklabels()[-1].set_color('#C44E52')

plt.tight_layout()
plt.savefig('d:\\UG\\Call intrusion\\model_performance_comparison.png', dpi=300)
print("Plot saved successfully.")

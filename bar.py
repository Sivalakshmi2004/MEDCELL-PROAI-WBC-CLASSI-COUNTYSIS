import matplotlib.pyplot as plt
import os
import numpy as np

# Calculate class distribution
class_names = os.listdir(r'C:\Users\Administrator\Desktop\proj\Dataset_Balanced')  # Adjust the directory path if necessary
class_counts = [len(os.listdir(os.path.join(r'C:\Users\Administrator\Desktop\proj\Dataset_Balanced', class_name))) for class_name in class_names]

# Generate random colors for each bar
colors = np.random.rand(len(class_names), 3)  # Random RGB colors for each bar

# Plot class distribution with colored bars
plt.bar(class_names, class_counts, color=colors)
plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.title('Class Distribution in Test Dataset')
plt.xticks(rotation=45)
plt.show()

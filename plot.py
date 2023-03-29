import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib as mpl
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler


rank = np.load("performance_history_rank_50.npy")
rank_random = np.load("performance_history_random_rank_50.npy")
comm = np.load("performance_history_committee_50.npy")
comm_random = np.load("performance_history_random_committee_50.npy")


plt.figure(figsize=(8, 8))

plt.plot(np.arange(50, 100, 5), rank, color='red', label="Ranked Batch")
plt.plot(np.arange(50, 100, 5), rank_random, color='red', linestyle='dashed', label="Random 5-step")
plt.plot(np.arange(50, 101, 1), comm, color='blue', label="QBC")
plt.plot(np.arange(50, 100, 1), comm_random, color='blue', linestyle='dashed', label="Random 1-step")

plt.legend()
plt.title('Performance History')
plt.xticks(np.arange(50, 101, 5))
plt.xlabel("Training set size")
plt.ylabel("F1-Score")
plt.savefig("performance_comparison_50_2.png")



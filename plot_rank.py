import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib as mpl
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler


RANDOM_STATE_SEED = 42
QUERY_SIZE = 20
X = np.load("AktivML/image_data_gray.npy")
y = np.load("AktivML/labels.npy")
performance_history = np.load("performance_history_rank.npy")
query_history = np.load("query_history_rank.npy")
query_history = query_history.reshape(-1, 2)

rus = RandomUnderSampler(random_state=0)
X_resampled, y_resampled = rus.fit_resample(X.reshape(4000,-1), y.reshape(4000,-1))
X_resampled, y_resampled = np.reshape(X_resampled, (2000,80,80,1)), np.reshape(y_resampled, (2000,1))


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.20, random_state=42)

# Get PCA from data
X_flat = np.array([img.flatten() for img in X])
# X_flat = np.array([img.flatten() for img in X_resampled])

pca = PCA(n_components=2, random_state=RANDOM_STATE_SEED)
transformed_X = pca.fit_transform(X=X_flat)
x_component, y_component = transformed_X[:, 0], transformed_X[:, 1]

# Plot
plt.figure(figsize=(8, 8), dpi=120)
plt.scatter(x=x_component, y=y_component, c=y, s=10, alpha=8/10)

# Text (For the first query batch)
for i, coord in enumerate(query_history[:QUERY_SIZE]):
    plt.text(x=coord[0], y=coord[1], s=f"{i+1}", 
    fontdict=dict(color="black", size=10), bbox=dict(boxstyle="round", fc="w"))



plt.title('PCA transformation of data - Rank Overlay')
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.savefig("pca_overlay_rank.png")



# Plot performance
plt.figure(figsize=(8, 8))
plt.plot(performance_history)

plt.title('Performance History')
plt.xticks(np.arange(0, len(performance_history)))
plt.xlabel("Query")
plt.ylabel("Score")
plt.savefig("performance_history_rank.png")






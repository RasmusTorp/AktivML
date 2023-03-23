import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib as mpl
import matplotlib.pyplot as plt

RANDOM_STATE_SEED = 42
X = np.load("AktivML/image_data_gray.npy")
y = np.load("AktivML/labels.npy")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Get PCA from data
X_flat = np.array([img.flatten() for img in X])
pca = PCA(n_components=2, random_state=RANDOM_STATE_SEED)
transformed_X = pca.fit_transform(X=X_flat)
x_component, y_component = transformed_X[:, 0], transformed_X[:, 1]

# Plot
plt.figure(figsize=(8, 8), dpi=120)
plt.scatter(x=x_component, y=y_component, c=y, s=10, alpha=8/10)
plt.title('PCA transformation of data')
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.savefig("pca.png")


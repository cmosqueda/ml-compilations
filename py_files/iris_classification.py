import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Load the dataset
df = pd.read_csv('datasets/iris.csv')

# Drop the 1st, 2nd, and 6th columns
df = df.drop(df.columns[[0, 1, 5]], axis=1)

# The last remaining column (previously the 5th) is the classification
X = df.iloc[:, :-1].values  # Features (all except the last column)
y = df.iloc[:, -1].values   # Target (classification)

# Encode classification labels (Iris species)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Define class names manually
class_names = ["setosa", "versicolor", "virginica"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN model with k=10
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

# New instance for prediction
new_instance = np.array([[3.6, 2.0]])  # Petal length = 3.6, Petal width = 2.0
predicted_class = knn.predict(new_instance)
predicted_label = class_names[predicted_class[0]]

print(f"Predicted classification for (3.6, 2.0): {predicted_label}")

# Visualization (only if X has exactly 2 features)
def plot_knn(X, y, model, class_names, new_instance=None):
    h = 0.02  # Step size in the mesh

    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict on mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)

    # Plot new instance
    if new_instance is not None:
        plt.scatter(new_instance[0, 0], new_instance[0, 1], c='red', edgecolors='black', 
                    s=150, label="New Instance", marker='X')

    # Custom legend
    handles, _ = scatter.legend_elements()
    plt.legend(handles + [plt.Line2D([0], [0], marker='X', color='w', markersize=10, 
                                     markerfacecolor='red', label="New Instance")], 
               class_names + ["New Instance"], title="Classes")

    plt.xlabel("Petal Length")
    plt.ylabel("Petal Width")
    plt.title("KNN Decision Boundary (k=10)")
    plt.show()

# Ensure the feature set has exactly 2 features for visualization
if X.shape[1] == 2:
    plot_knn(X_train, y_train, knn, class_names, new_instance)
else:
    print(f"Visualization requires exactly 2 features. Your dataset has {X.shape[1]} features.")

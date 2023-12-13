import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io, transform
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

labels_words = {
    0: 'uneven road',
    1: 'Hump',
    2: 'Slippary road',
    3:'Left Curve'
    # Add more mappings as needed
}
root_directory = 'Training'
classes = [class_label for class_label in os.listdir(root_directory)
           if os.path.isdir(os.path.join(root_directory, class_label))][:4]

images = []
labels = []

target_shape = (32, 32)  # Set your desired target shape
scaler = StandardScaler()
for class_label in classes:
    class_path = os.path.join(root_directory, class_label)

    # Add a check to ensure class_path is a directory
    if not os.path.isdir(class_path):
        print(f"Skipped non-directory: {class_path}")
        continue

    for image_file in os.listdir(class_path):
        image_path = os.path.join(class_path, image_file)

        # Skip non-image files
        if image_file.lower().endswith(('.ppm', '.png', '.jpg', '.jpeg', '.gif')):
            # Read the image
            image = io.imread(image_path)


            image = transform.resize(image, target_shape, anti_aliasing=True)

            flattened_image = image.flatten()


            flattened_image = flattened_image.reshape(-1, 1)
            flattened_image = scaler.fit_transform(flattened_image)


            flattened_image = flattened_image.flatten()
            images.append(flattened_image)
            labels.append(int(class_label))


pca = PCA(n_components=32)
# Convert lists to NumPy arrays
images = np.array(images)
labels = np.array(labels)
reduced_images=pca.fit_transform(images)
labels_as_words = [labels_words[label] for label in labels]

x=reduced_images
y=labels
print(x.shape)
print(y.shape)
X_train, X_test = train_test_split(x, test_size=0.2, random_state=42)
kmeans_train = KMeans(n_clusters=4,n_init=10, random_state=42)
kmeans_train.fit(X_train)
cluster_assignments_train = kmeans_train.labels_
kmeans_test = KMeans(n_clusters=4,n_init=10, random_state=42)
kmeans_test.fit(X_test)
cluster_assignments_test = kmeans_test.labels_
plt.scatter(X_train[:, 0], X_train[:, 1], c=cluster_assignments_train, cmap='viridis', s=50)
plt.scatter(kmeans_train.cluster_centers_[:, 0], kmeans_train.cluster_centers_[:, 1], marker='X', s=200, color='red', label='Centroids')

plt.title(f'KMeans Clustering with {4} Clusters (Training Data)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# Visualize the clusters in testing data (assuming 2D data for simplicity)
plt.scatter(X_test[:, 0], X_test[:, 1], c=cluster_assignments_test, cmap='viridis', s=50)
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# df=pd.read_csv("Training/00000/GT-00000.csv",sep=";")
# print(df.head())
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# model = LogisticRegression(C=1.0,max_iter=1000)
# model.fit(X_train_scaled, y_train)
#
# print("############### cv scores ###############")
# cv_scores = cross_val_score(model, x, y, cv=5)
# print("Cross-Validation Scores:", cv_scores)
# print(f"Mean CV Score: {np.mean(cv_scores):.4f}")
# print(f"Standard Deviation of CV Scores: {np.std(cv_scores):.4f}")
# print("############### accuracy ###############")
# y_pred = model.predict(X_test_scaled)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy on the test set: {accuracy:.4f}")
# print("############### confusion_matrix ###############")
# conf_matrix = confusion_matrix(y_test, y_pred)
# print("Confusion Matrix:")
# print(conf_matrix)

# print("############### ROC ###############")
# positive_class_label = 3  # 'Left Curve'
#
# # Convert labels to binary (1 for positive class, 0 for others)
# binary_labels = np.where(labels == positive_class_label, 1, 0)
# y_prob = model.predict_proba(X_test_scaled)[:, 1]  # Probability of the positive class
# fpr, tpr, thresholds = roc_curve(y_test, y_prob)
# roc_auc = auc(fpr, tpr)
#
# # Plot ROC Curve
# plt.figure(figsize=(8, 8))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc="lower right")
# plt.show()
#


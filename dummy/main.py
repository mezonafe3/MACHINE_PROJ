import os
import numpy as np
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from skimage import io, transform
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.preprocessing import StandardScaler

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

            # Resize the image to the target shape
            image = transform.resize(image, target_shape, anti_aliasing=True)
            # image=scaler.fit_transform(image)
            # flattened_image = image.flatten()
            flattened_image = image.flatten()

            # Reshape to 1D array before applying StandardScaler
            flattened_image = flattened_image.reshape(-1, 1)
            flattened_image = scaler.fit_transform(flattened_image)

            # Reshape back to 1D
            flattened_image = flattened_image.flatten()
            images.append(flattened_image)
            labels.append(int(class_label))  # Assuming class labels are represented as integers



# Convert lists to NumPy arrays
pca = PCA(n_components=32)
# Convert lists to NumPy arrays
images = np.array(images)
labels = np.array(labels)
reduced_images=pca.fit_transform(images)

labels_as_words = [labels_words[label] for label in labels]
# print(images.shape)  # Print the shape of the resulting array
# print(labels.data)
# print(labels[54])
# selected_labels = labels[labels == 4]

# Print the selected labels
# print(len(selected_labels))
# x=5
# fig, axes = plt.subplots(1, x, figsize=(15, 5))
#
# for i in range(x):
#     axes[i].imshow(images[i])
#     axes[i].set_title(f'Class {labels_as_words[i]}')
#     axes[i].axis('off')
#
# plt.show()
#trainning
x=reduced_images
y=labels
model = LogisticRegression(C=1.0,max_iter=1000)
model.fit(x,y)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
cv_scores = cross_val_score(model, x, y, cv=5)
print("Cross-Validation Scores:", cv_scores)

# Print the mean and standard deviation of the cross-validation scores
print(f"Mean CV Score: {np.mean(cv_scores):.4f}")
print(f"Standard Deviation of CV Scores: {np.std(cv_scores):.4f}")

# test_image_path = 'Training/00000/01153_00002.ppm'
# target_shape = (32, 32)  # Set your desired target shape
#
# # Read the image
# test_image = io.imread(test_image_path)
#
# # Resize the image to the target shape
# test_image_resized = transform.resize(test_image, target_shape, anti_aliasing=True)
#
# # Flatten the image
# flattened_test_image = test_image_resized.flatten()
#
# # Reshape to 2D array before applying StandardScaler
# flattened_test_image = flattened_test_image.reshape(1, -1)
#
# # Ensure that the scaler is expecting the same number of features as the test image
# if flattened_test_image.shape[1] != scaler.mean_.shape[0]:
#     raise ValueError("Number of features does not match the trained scaler")
#
# # Use the same scaler that you used during training
# flattened_test_image_scaled = scaler.transform(flattened_test_image)
#
# # Make a prediction
# y_pred = model.predict(flattened_test_image_scaled)

# Print the predicted class
# print(f"Predicted class: {y_pred}")
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy on the test set: {accuracy}")











































# Plot decision boundary
fig, ax = plt.subplots(figsize=(8, 8))

# Plot decision regions
plot_decision_regions(X=x, y=y, clf=model, legend=2, ax=ax)

# Add axes labels
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Add a title
plt.title('Logistic Regression Decision Boundaries')

# Add legend
plt.legend()

plt.show()


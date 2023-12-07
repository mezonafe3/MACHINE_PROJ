import os
from PIL import Image
import numpy as np
import pandas as pd
from mlxtend.plotting import plot_decision_regions
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
scaler = StandardScaler()

root_directory = 'Training'
classes = [class_label for class_label in os.listdir(root_directory)
           if os.path.isdir(os.path.join(root_directory, class_label))][:3]

images = []
labels = []

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
           image=Image.open(image_path)
           image=image.resize((30,30))
           image=np.array(image)

           images.append(image)
           labels.append(class_label)

# print(len(images))
# print(labels[30])
# plt.imshow(images[30])
# plt.axis("off")
# plt.show()
# print(images.shape)
# print(images.shape)
# print(labels.shape)




# print(test_labels)
images=np.array(images)
labels=np.array(labels)
path='classes.txt'
file=open(path)
cls={}
for i in file:
 data=i.split('-')
 cls.update({data[0]:data[1]})

# X, y = make_classification(n_samples=153, n_features=10, n_classes=3, random_state=42)

X_train,X_test,y_train,y_test=train_test_split(images,labels,test_size=0.2,random_state=4,shuffle=True)
# print(X_train.shape)
# print(cls[str(labels[36])])
# print(y_train)

unique_classes = np.unique(labels)
num_classes = len(unique_classes)

y_train_one_hot=np.eye(num_classes,dtype='int')[y_train.astype('int')]
y_test_one_hot=np.eye(num_classes,dtype='int')[y_test.astype('int')]
y_train_flat = np.argmax(y_train_one_hot, axis=1)
y_test_flat = np.argmax(y_test_one_hot, axis=1)

X_train_reduced=np.reshape(X_train, (X_train.shape[0], -1))
X_test_reduced = np.reshape(X_test, (X_test.shape[0], -1))

X_train_scaled = scaler.fit_transform(X_train_reduced)
X_test_scaled = scaler.transform(X_test_reduced)

test_path='Testing'
test_df=pd.read_csv('Testing/test.csv',sep=';')
test_images=test_df['Filename'].values
test_labels=test_df['ClassId'].values
images_for_test=[]
for img in test_images:
    image=Image.open(test_path+'\\'+img)
    image=image.resize((30,30))
    image=np.array(image)
    images_for_test.append(image)
images_for_test=np.array(images_for_test)
labels_for_test=np.array(test_labels)



labels_for_test_he=np.eye(num_classes,dtype='int')[labels_for_test.astype('int')]
labels_for_test_he_flat = np.argmax(labels_for_test_he, axis=1)###########


images_for_test_reduced=np.reshape(images_for_test, (images_for_test.shape[0], -1))
images_for_test_scaled = scaler.fit_transform(images_for_test_reduced)#########
# print(y_train_one_hot.shape)
# print(y_test)
model=LogisticRegression(multi_class='ovr')
model.fit(X_train_scaled,y_train_flat)
X_test_reduced = np.reshape(X_test, (X_test.shape[0], -1))
y_pred = model.predict(X_test_scaled)
l_n_pred=model.predict(images_for_test_scaled)
accuracy = accuracy_score(y_test_flat,y_pred)
accuracy_n_data = accuracy_score(labels_for_test_he_flat,l_n_pred)
print("accuracy_n_data:", accuracy_n_data)
# print('###############ACCURACY#################')
# print("Accuracy:", accuracy)
# print('###############Confusion Matrix#################')
# class_labels=['uneven road','Hump','Slippary road']
# cm = confusion_matrix(y_test_flat, y_pred)
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
# plt.title("Confusion Matrix")
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.show()
#
print('###############ACCURACY#################')
print("Accuracy:", accuracy)
print('###############Confusion Matrix#################')
class_labels=['uneven road','Hump','Slippary road']
cm = confusion_matrix(labels_for_test_he_flat, l_n_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
i=10
data_one=np.expand_dims(images_for_test[i],axis=0)
test_image_reshape=np.reshape(data_one, (data_one.shape[0], -1))
test_prediction=model.predict(test_image_reshape)
actual_value = test_prediction[0]
plt.imshow(images_for_test[i])
plt.title(cls[str(actual_value)], fontsize=16, fontweight='bold')
plt.axis("off")
plt.show()
# y_prob = model.predict_proba(X_test_scaled)
#
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
#
# for i in range(num_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_test_one_hot[:, i], y_prob[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])
#
# fpr["micro"], tpr["micro"], _ = roc_curve(y_test_one_hot.ravel(), y_prob.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#
# plt.figure(figsize=(10, 8))
#
# plt.plot(fpr["micro"], tpr["micro"], label=f'Micro-average ROC curve (area = {roc_auc["micro"]:.2f})',
#          color='deeppink', linestyle=':', linewidth=4)
#
# for i in range(num_classes):
#     plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {cls[str(i)]} (area = {roc_auc[i]:.2f})')
#
# plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc="lower right")
# plt.show()

# # AUC-ROC Curve
# print('###############AUC-ROC Curve#################')
# y_prob = model.predict_proba(X_test_reduced)
# roc_auc = roc_auc_score(y_test_flat, y_prob, multi_class='ovr')
# print("AUC-ROC Score:", roc_auc)
#
# # Plot ROC curves
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
#
# for i in range(num_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_test_one_hot[:, i], y_prob[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])
#
# plt.figure(figsize=(8, 6))
# for i in range(num_classes):
#     plt.plot(fpr[i], tpr[i], label=f'{cls[str(i)]} (AUC = {roc_auc[i]:.2f})')
#
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc='lower right')
# plt.show()
#
#



# fig, ax = plt.subplots(figsize=(8, 8))
# plot_decision_regions(X=X_train_reduced, y=y_train_flat, clf=model, legend=2, ax=ax)
# plt.xlabel('images')
# plt.ylabel('classes')
# plt.title('Logistic Regression Decision Boundaries')
# plt.legend()
# plt.show()
# Compute ROC curve and AUC
# fpr, tpr, thresholds = roc_curve(y_test_flat, y_pred)
# roc_auc = auc(fpr, tpr)
#
# # Plot ROC curve
# plt.figure(figsize=(8, 8))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc='lower right')
# plt.show()
# auc_scores = []
# for i in range(3):
#     auc_scores.append(roc_auc_score(y_test_flat == i, y_pred[:, i]))
#
# # Print AUC scores for each class
# for i, auc in enumerate(auc_scores):
#     print(f'AUC for Class {i}: {auc:.2f}')
#
# # Optionally, you can average the AUC scores for all classes
# average_auc = np.mean(auc_scores)
# print(f'Average AUC: {average_auc:.2f}')


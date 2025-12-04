

# --- 1. SETUP & DEPENDENCIES ---
import os
import sys

# Install required libraries
os.system('pip install tensorflow pandas numpy matplotlib seaborn scikit-learn opencv-python-headless gdown')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
import tensorflow as tf
import zipfile
import gdown
import shutil
from PIL import Image
from google.colab import drive
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

print("Libraries successfully installed and imported.")

# --- 2. DATASET LOADING  ---
file_id = '17RjAev495D2uh2sHwIO2qYDUR4B8POq7'
zip_filename = 'Mango_Dataset.zip'
extract_path = '/content/dataset_unzipped'


def is_valid_zip(path):
    return os.path.exists(path) and zipfile.is_zipfile(path)


if not is_valid_zip(zip_filename):
    print("Attempting direct download from Google Drive...")
    try:
        gdown.download(id=file_id, output=zip_filename, quiet=False)
    except Exception as e:
        print(f"Direct download warning: {e}")

if not is_valid_zip(zip_filename):
    print("\nDirect download failed. Mounting Google Drive to find file...")
    if os.path.exists(zip_filename): os.remove(zip_filename) # Clean bad file
    
    drive.mount('/content/drive')
    
    possible_paths = [
        '/content/drive/MyDrive/Mango_Dataset/Mango_Dataset.zip', 
        '/content/drive/MyDrive/Mango_Dataset.zip',
        f'/content/drive/MyDrive/{zip_filename}' 
    ]
    
    found_path = None
    for path in possible_paths:
        if os.path.exists(path):
            found_path = path
            break
            
    if found_path:
        print(f"Found file at: {found_path}")
        shutil.copy(found_path, zip_filename)
    else:
        print("ERROR: Could not find 'Mango_Dataset.zip' in your Drive.")
        print("Please ensure the file is uploaded to your Google Drive.")
        raise FileNotFoundError("Dataset zip not found.")

print(f"Extracting {zip_filename}...")
try:
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extraction complete.")
except zipfile.BadZipFile:
    raise Exception("CRITICAL: The file downloaded is not a valid zip file. Please checks permissions.")

dataset_dir = os.path.join(extract_path, 'Dataset', 'Augmented data')
if not os.path.exists(dataset_dir):
    for root, dirs, files in os.walk(extract_path):
        if 'Augmented data' in dirs:
            dataset_dir = os.path.join(root, 'Augmented data')
            break
print(f"Dataset directory found at: {dataset_dir}")

# --- 3. DATA PREPROCESSING ---
data = []
labels = []

print("Reading image files...")
for fold in os.listdir(dataset_dir):
    filepaths = os.path.join(dataset_dir, fold)
    if os.path.isdir(filepaths):
        for file in os.listdir(filepaths):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                data.append(os.path.join(filepaths, file))
                labels.append(fold)

# Create DataFrame
dataset = pd.concat([pd.Series(data, name='paths'), pd.Series(labels, name='labels')], axis=1)

training_data, temp_validation_data = train_test_split(dataset, test_size=0.2, shuffle=True, random_state=42, stratify=dataset['labels'])
validation_data, test_data = train_test_split(temp_validation_data, test_size=0.05, shuffle=True, random_state=42, stratify=temp_validation_data['labels'])

print(f"Training Samples: {len(training_data)}")
print(f"Validation Samples: {len(validation_data)}")
print(f"Test Samples: {len(test_data)}")

# Image Generators
img_shape = (224, 224)
batch_size = 32
def return_image(image): return image

train_gen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=return_image)
validation_gen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=return_image)

training_data_gen = train_gen.flow_from_dataframe(
    training_data, x_col="paths", y_col="labels",
    target_size=img_shape, class_mode='categorical', batch_size=batch_size
)
validation_data_gen = validation_gen.flow_from_dataframe(
    validation_data, x_col="paths", y_col="labels",
    target_size=img_shape, class_mode='categorical', batch_size=batch_size
)

# --- 4. MODEL 1: CUSTOM CNN ---
print("\n--- Building Custom CNN Model ---")
inputs = tf.keras.Input(shape=(224,224,3))
x = tf.keras.layers.Conv2D(filters=32, kernel_size=(4,4), strides=2)(inputs)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.MaxPool2D(pool_size=(4,4), name='max_pooling2d_1')(x) # Naming for GradCAM later
x = tf.keras.layers.Conv2D(filters=32, kernel_size=(2,2), strides=1)(x)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.MaxPool2D(pool_size=(3,3))(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(units=512)(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.25)(x)
x = tf.keras.layers.Dense(units=512)(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.1)(x)
outputs = tf.keras.layers.Dense(units=8, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Callbacks
checkpoint = ModelCheckpoint("model.h5", monitor="val_loss", mode="min", save_best_only=True, verbose=1)
earlystopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.2, min_lr=0.00000001)

print("Starting Training...")
history = model.fit(
    training_data_gen,
    validation_data=validation_data_gen,
    epochs=20, 
    callbacks=[checkpoint, earlystopping, learning_rate_reduction]
)

# --- 5. VISUALIZATION (PLOTS) ---
training_acc = history.history['accuracy']
training_loss = history.history['loss']
validation_acc = history.history['val_accuracy']
validation_loss = history.history['val_loss']

index_loss = np.argmin(validation_loss)
val_lowest = validation_loss[index_loss]
index_acc = np.argmax(validation_acc)
acc_highest = validation_acc[index_acc]

epochs_range = [i+1 for i in range(len(training_acc))]
loss_label = f'best epoch= {str(index_loss + 1)}'
acc_label = f'best epoch= {str(index_acc + 1)}'

plt.figure(figsize= (20, 8))
plt.style.use('fivethirtyeight')

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(epochs_range, training_loss, 'r', label= 'Training loss')
plt.plot(epochs_range, validation_loss, 'g', label= 'Validation loss')
plt.scatter(index_loss + 1, val_lowest, s= 150, c= 'blue', label= loss_label)
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs_range, training_acc, 'r', label= 'Training Accuracy')
plt.plot(epochs_range, validation_acc, 'g', label= 'Validation Accuracy')
plt.scatter(index_acc + 1 , acc_highest, s= 150, c= 'blue', label= acc_label)
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# --- 6. TESTING & CONFUSION MATRIX ---
plt.style.use('default')
test_gen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=return_image, rescale=1./255)
test_data_gen = test_gen.flow_from_dataframe(
    test_data, x_col="paths", y_col="labels", shuffle=False, # Shuffle False for Correct CM
    class_mode='categorical', target_size=img_shape, batch_size=batch_size
)

preds = model.predict(test_data_gen)
y_pred = np.argmax(preds, axis=1)
classes = list(test_data_gen.class_indices.keys())

cm = confusion_matrix(test_data_gen.classes, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, square=True, xticklabels=classes, yticklabels=classes)
plt.xlabel('Predictions')
plt.ylabel('True values')
plt.title('Confusion Matrix of the CNN model')
plt.show()

print(classification_report(test_data_gen.classes, y_pred, target_names=classes))

# --- 7. GRAD-CAM VISUALIZATION ---
def get_img_array(img_path, img_shape):
    img = tf.keras.utils.load_img(img_path, target_size=img_shape) 
    array = tf.keras.utils.img_to_array(img) / 255.0    
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)    
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(img_path, heatmap, alpha=0.4):
    img = tf.keras.utils.load_img(img_path)
    img = tf.keras.utils.img_to_array(img) 
    heatmap = np.uint8(255 * heatmap)
    jet = matplotlib.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)
    return superimposed_img

print("Generating Grad-CAM...")
last_conv_layer_name = "max_pooling2d_1" # Name matched to model definition above
fig, ax = plt.subplots(2, 6, figsize=(16,12))

for i in range(6):
    img_path = test_data.iloc[i, 0]
    img_array = get_img_array(img_path, img_shape=img_shape)
    
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    
    # Display
    ax[0,i].matshow(heatmap)
    ax[1,i].imshow(display_gradcam(img_path, heatmap))

plt.tight_layout()
fig.text(0.5, 0.92, "GRAD-CAM of 6 samples of test set", ha='center', fontsize=20)
fig.text(0.5, 0.49, "Superimposition of heatmap and samples", ha='center', fontsize=20)
plt.show()


# --- 8. VGG16 FEATURE EXTRACTION + KNN/SVM ---
print("\n--- Starting VGG16 Feature Extraction ---")

def load_dataset_folder(folder, img_size=(224, 224)):
    images = []
    labels = []
    
    print("Loading images for Feature Extraction...")
    for label in os.listdir(folder):
        file_path = os.path.join(folder, label)
        if os.path.isdir(file_path):
            files = os.listdir(file_path)
            # Limit per class to balance speed/accuracy
            for file in files[:300]: 
                img_path = os.path.join(file_path, file)
                try:
                    image = cv2.imread(img_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, img_size)
                    images.append(image)
                    labels.append(label)
                except: pass
    return np.array(images), np.array(labels)

X_vgg, y_vgg = load_dataset_folder(dataset_dir)
print(f"Loaded {len(X_vgg)} images for Transfer Learning.")

X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(X_vgg, y_vgg, test_size=0.2, random_state=42)

# VGG Model
base_model = tf.keras.applications.VGG16(include_top=False, weights="imagenet", input_shape=(224,224,3))
base_model.trainable = False

deep_feat_model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(), # Better than Flatten for variable input
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(256, activation="linear")
])

print("Extracting Deep Features...")
train_features = deep_feat_model.predict(X_train_v)
test_features = deep_feat_model.predict(X_test_v)

# Flatten
training_flatten = train_features.reshape(train_features.shape[0], -1)
test_flatten = test_features.reshape(test_features.shape[0], -1)

# --- KNN ---
print("\n--- Training KNN ---")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(training_flatten, y_train_v)
y_pred_knn = knn_model.predict(test_flatten)

print(classification_report(y_test_v, y_pred_knn))

plt.figure()
sns.heatmap(confusion_matrix(y_test_v, y_pred_knn), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix: KNN')
plt.show()

# --- SVM ---
print("\n--- Training SVM ---")
svm_model = SVC()
svm_model.fit(training_flatten, y_train_v)
y_pred_svm = svm_model.predict(test_flatten)

print(classification_report(y_test_v, y_pred_svm))

plt.figure()
sns.heatmap(confusion_matrix(y_test_v, y_pred_svm), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix: SVM')
plt.show()

print("\nAll Tasks Completed Successfully.")
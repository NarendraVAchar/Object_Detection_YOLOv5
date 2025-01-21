import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Preprocessing with augmentation
datagen = ImageDataGenerator(
    rescale=1./255,               # Normalize pixel values
    rotation_range=15,            # Random rotation
    width_shift_range=0.1,        # Random horizontal shift
    height_shift_range=0.1,       # Random vertical shift
    validation_split=0.2          # Split 20% for validation
)

# Training data
train_data = datagen.flow_from_directory(
    'dataset/train',              # Path to your training data
    target_size=(128, 128),       # Resize images
    batch_size=32,
    class_mode='categorical',     # Multi-class classification
    subset='training'             # Use for training set
)

# Validation data
val_data = datagen.flow_from_directory(
    'dataset/val',                # Path to your validation data
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',     # Multi-class classification
    subset='validation'           # Use for validation set
)

# Multi-class classification model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(train_data.class_indices), activation='softmax')  # 4 output units for 4 classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_data, validation_data=val_data, epochs=50)

# Save the trained model
model.save('device_status_model.keras')

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
import numpy as np
import random
import os

# Set random seed for reproducibility
def set_random_seed(seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_random_seed()

class GestureRecognitionModel:
    def __init__(self, input_shape=(64, 64, 1), num_classes=2):
        print(f"Initializing GestureRecognitionModel with input_shape={input_shape}, num_classes={num_classes}")
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model =self.build_model()

    @staticmethod
    def inspect_and_visualize_dataset(dataset, num_samples=5):
        print("Inspecting and visualizing dataset...")
        for images, labels in dataset.take(1):  # Take one batch of data
            for i in range(num_samples):
                image = images[i].numpy().squeeze()  # Squeeze to remove the single color channel for visualization
                label = labels[i].numpy()
                print(f"Label: {label}")
                plt.imshow(image, cmap='gray')
                plt.title(f"Label: {label}")
                plt.show()

    def build_model(self):
        print("Building the model...")
        cnn_base = tf.keras.Sequential([
            tf.keras.layers.Input(shape=self.input_shape),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5)  # Added dropout for regularization
        ])

        inputs = tf.keras.Input(shape=self.input_shape)
        x = cnn_base(inputs)
        x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = tf.keras.layers.Dropout(0.5)(x)  # Added dropout for regularization
        outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)

        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print("Model built successfully.")
        return model

    def train(self, train_dataset, val_dataset, epochs=10, steps_per_epoch=100, validation_steps=50):
        print(f"Starting training for {epochs} epochs...")
        train_dataset = train_dataset.repeat()  # Repeat dataset indefinitely
        val_dataset = val_dataset.repeat()  # Repeat validation dataset indefinitely

        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps
        )
        print("Training completed.")
        return history

    def evaluate(self, test_dataset, steps=50):
        print("Evaluating the model...")
        test_dataset = test_dataset.repeat()  # Repeat the test dataset indefinitely
        test_loss, test_acc = self.model.evaluate(test_dataset, steps=steps)
        print(f"Evaluation completed. Test Loss: {test_loss}, Test Accuracy: {test_acc}")
        return test_loss, test_acc

    @staticmethod
    def plot_metrics(history):
        print("Plotting training and validation metrics...")
        fig, axs = plt.subplots(2, 1, figsize=(12, 8))

        # Plot training & validation accuracy values
        axs[0].plot(history.history['accuracy'])
        axs[0].plot(history.history['val_accuracy'])
        axs[0].set_title('Model accuracy')
        axs[0].set_ylabel('Accuracy')
        axs[0].set_xlabel('Epoch')
        axs[0].legend(['Train', 'Validation'], loc='upper left')

        # Plot training & validation loss values
        axs[1].plot(history.history['loss'])
        axs[1].plot(history.history['val_loss'])
        axs[1].set_title('Model loss')
        axs[1].set_ylabel('Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].legend(['Train', 'Validation'], loc='upper left')

        plt.tight_layout()
        plt.show()
        print("Metrics plotted successfully.")

    def save_model(self, model_path):
        print(f"Saving the model to {model_path}...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        print(f"Model saved to {model_path}.")

    def save_model_weights(self, weights_path):
        print(f"Saving the model weights to {weights_path}...")
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        self.model.save_weights(weights_path)
        print(f"Model weights saved to {weights_path}.")

    @classmethod
    def load_model(cls, model_path):
        print(f"Loading the model from {model_path}...")
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}.")
        # Create a new instance of the class without building a new model
        obj = cls.__new__(cls)
        obj.model = model
        return obj

    @classmethod
    def load_model_weights(cls, input_shape, num_classes, weights_path):
        print(f"Loading the model weights from {weights_path}...")
        # Initialize a new model instance
        obj = cls(input_shape, num_classes)
        # Load the weights
        obj.model.load_weights(weights_path)
        print(f"Model weights loaded from {weights_path}.")
        return obj

    def plot_model_structure(self, plot_path='model_structure.png'):
        print(f"Plotting the model structure to {plot_path}...")
        plot_model(self.model, to_file=plot_path, show_shapes=True, show_layer_names=True)
        print(f"Model structure plot saved to {plot_path}.")

    @staticmethod
    def parse_tfrecord_fn(example):
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        }
        example = tf.io.parse_single_example(example, feature_description)
        image = tf.io.parse_tensor(example['image'], out_type=tf.float32)
        image = tf.reshape(image, [64, 64, 1])  # Adjust shape for grayscale
        label = tf.cast(example['label'], tf.int64)  # Ensure label is int64
        return image, label

    @staticmethod
    def load_tfrecord_dataset(file_pattern, batch_size=50):
        dataset = tf.data.TFRecordDataset(tf.data.Dataset.list_files(file_pattern))
        dataset = dataset.map(GestureRecognitionModel.parse_tfrecord_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(1000)  # Add shuffling here
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

# Example usage:
if __name__ == "__main__":
    # Load datasets from TFRecord files
    print("Loading datasets from TFRecord files...")
    train_dataset = GestureRecognitionModel.load_tfrecord_dataset('datasets/train.tfrecord')
    val_dataset = GestureRecognitionModel.load_tfrecord_dataset('datasets/val.tfrecord')
    test_dataset = GestureRecognitionModel.load_tfrecord_dataset('datasets/test.tfrecord')

    # Inspect and visualize the TFRecord datasets to ensure correctness
    GestureRecognitionModel.inspect_and_visualize_dataset(train_dataset)

    # Initialize and train the model
    gesture_model = GestureRecognitionModel(input_shape=(64, 64, 1), num_classes=2)  # Updated to match grayscale input
    gesture_model.model = gesture_model.build_model()

    # Train the model
    history = gesture_model.train(train_dataset, val_dataset, epochs=10)

    # Evaluate the model
    test_loss, test_acc = gesture_model.evaluate(test_dataset, steps=50)
    print(f"Test accuracy: {test_acc}")

    # Plot training metrics
    gesture_model.plot_metrics(history)

    # Save the model
    model_dir = 'model'
    model_path = os.path.join(model_dir, 'gesture_recognition_model.h5')
    gesture_model.save_model(model_path)

    # Save model weights
    weights_dir = os.path.join(model_dir, 'weights')
    weights_path = os.path.join(weights_dir, 'gesture_recognition_weights.weights.h5')
    gesture_model.save_model_weights(weights_path)

    # Load the model
    loaded_gesture_model = GestureRecognitionModel.load_model(model_path)

    # Evaluate the loaded model
    test_loss_load, test_acc_load = loaded_gesture_model.evaluate(test_dataset, steps=50)
    print(f"Test accuracy (loaded model): {test_acc_load}")

    # Load the model with weights
    loaded_gesture_model_with_weights = GestureRecognitionModel.load_model_weights(
        input_shape=(64, 64, 1),
        num_classes=2,
        weights_path=weights_path
    )

    # Evaluate the model with loaded weights
    test_loss_weights, test_acc_weights = loaded_gesture_model_with_weights.evaluate(test_dataset, steps=50)
    print(f"Test accuracy (model with loaded weights): {test_acc_weights}")
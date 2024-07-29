import tensorflow as tf
import os
from sklearn.model_selection import train_test_split

class DatasetPipeline:
    def __init__(self, output_path, target_shape=(64, 64), selected_gestures=None, validation_split=0.2,
                 test_split=0.1, epochs=2):
        self.output_path = output_path
        self.target_shape = target_shape + (1,)  # Adjust target_shape to include the grayscale channel
        self.selected_gestures = selected_gestures if selected_gestures else []
        self.validation_split = validation_split
        self.test_split = test_split
        self.epochs = epochs

    def load_image(self, image_path, label):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=1)  # Read as grayscale
        image = tf.image.resize(image, self.target_shape[:2])
        label = tf.cast(label, tf.int64)  # Ensure label is int64
        return image, label

    def augment_image(self, image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.image.random_crop(image, size=[int(self.target_shape[0] * 0.9), int(self.target_shape[1] * 0.9), 1])
        image = tf.image.resize(image, self.target_shape[:2])
        return image, label

    def load_dataset(self):
        print("Loading dataset...")
        image_paths = []
        labels = []
        class_names = os.listdir(self.output_path)
        for class_name in class_names:
            if class_name not in self.selected_gestures:
                continue
            label = self.selected_gestures.index(class_name)
            class_path = os.path.join(self.output_path, class_name)
            for image_file in os.listdir(class_path):
                image_paths.append(os.path.join(class_path, image_file))
                labels.append(label)

        image_paths = tf.constant(image_paths)
        labels = tf.constant(labels, dtype=tf.int64)  # Ensure labels are int64
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        dataset = dataset.map(self.load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        print(f"Dataset loaded with {len(image_paths)} images.")
        return dataset

    def prepare_dataset(self):
        dataset = self.load_dataset()

        # Print the size of the dataset before augmentation
        dataset_size_before_augmentation = len(list(dataset.as_numpy_iterator()))
        print(f"Dataset size before augmentation: {dataset_size_before_augmentation}")

        # Create an augmented dataset
        augmented_dataset = dataset.map(self.augment_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Combine original and augmented datasets
        combined_dataset = dataset.concatenate(augmented_dataset)

        # Print the size of the dataset after augmentation
        dataset_size_after_augmentation = len(list(combined_dataset.as_numpy_iterator()))
        print(f"Dataset size after augmentation: {dataset_size_after_augmentation}")

        # Convert combined dataset to a list of (image, label) pairs
        combined_data = list(combined_dataset.as_numpy_iterator())
        combined_images, combined_labels = zip(*combined_data)

        # Perform stratified splitting
        train_val_images, test_images, train_val_labels, test_labels = train_test_split(
            combined_images, combined_labels, test_size=self.test_split, stratify=combined_labels, random_state=42)
        train_images, val_images, train_labels, val_labels = train_test_split(
            train_val_images, train_val_labels, test_size=self.validation_split / (1 - self.test_split), stratify=train_val_labels, random_state=42)

        train_dataset = tf.data.Dataset.from_tensor_slices((list(train_images), list(train_labels)))
        val_dataset = tf.data.Dataset.from_tensor_slices((list(val_images), list(val_labels)))
        test_dataset = tf.data.Dataset.from_tensor_slices((list(test_images), list(test_labels)))

        batch_size = 50  # Change batch size to 50
        train_dataset = (train_dataset.shuffle(1000).
                         repeat(self.epochs).
                         batch(batch_size).
                         prefetch(tf.data.experimental.AUTOTUNE))
        val_dataset = val_dataset.repeat(self.epochs).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        return train_dataset, val_dataset, test_dataset

    @staticmethod
    def save_tfrecord(dataset, filename):
        def _serialize_example(image, label):
            image = tf.io.serialize_tensor(image)
            feature = {
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.numpy()])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label.numpy())]))
            }
            example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
            return example_proto.SerializeToString()

        with tf.io.TFRecordWriter(filename) as writer:
            for batch_images, batch_labels in dataset:
                for i in range(batch_images.shape[0]):
                    example = _serialize_example(batch_images[i], batch_labels[i])
                    print(f"Serializing image with shape: {batch_images[i].shape} and label: {batch_labels[i]}")
                    writer.write(example)

    def save_datasets(self, train_dataset, val_dataset, test_dataset):
        if not os.path.exists('datasets'):
            os.makedirs('datasets')
        else:
            print("Datasets directory already exists. Skipping creation.")

        self.save_tfrecord(train_dataset, 'datasets/train.tfrecord')
        self.save_tfrecord(val_dataset, 'datasets/val.tfrecord')
        self.save_tfrecord(test_dataset, 'datasets/test.tfrecord')
        print("Datasets saved to TFRecord files.")

    @staticmethod
    def inspect_dataset(dataset, num_samples=5):
        print("Inspecting dataset...")
        for images, labels in dataset.take(num_samples):
            for i in range(len(images)):
                image = images[i].numpy()
                label = labels[i].numpy()
                print(f"Label: {label}")
                print(f"Label shape: {labels.shape}")
                print(f"Image shape: {images.shape}")
                print(f"Image array:\n{image}")

    @staticmethod
    def compare_datasets(original_dataset, tfrecord_dataset, num_samples=5):
        print("Comparing datasets...")
        original_iterator = iter(original_dataset.unbatch().take(num_samples))
        tfrecord_iterator = iter(tfrecord_dataset.unbatch().take(num_samples))

        for _ in range(num_samples):
            try:
                original_image, original_label = next(original_iterator)
            except StopIteration:
                break
            try:
                tfrecord_image, tfrecord_label = next(tfrecord_iterator)
            except StopIteration:
                break

            print(f"Original image shape: {original_image.shape}, TFRecord image shape: {tfrecord_image.shape}")
            print(f"Original label: {original_label}, TFRecord label: {tfrecord_label}")
            assert original_image.shape == tfrecord_image.shape, "Image shapes do not match!"
            assert int(original_label) == int(tfrecord_label), "Labels do not match!"

        print("Dataset comparison completed successfully. The datasets match!")

    @staticmethod
    def parse_tfrecord_fn(example):
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),  # Keep label as int64
        }
        example = tf.io.parse_single_example(example, feature_description)
        image = tf.io.parse_tensor(example['image'], out_type=tf.float32)
        image = tf.reshape(image, [64, 64, 1])  # Adjust shape for grayscale
        label = tf.cast(example['label'], tf.int64)  # Ensure label is int64
        return image, label

    @staticmethod
    def load_tfrecord_dataset(file_pattern, batch_size=50):  # Updated batch size to 50
        dataset = tf.data.TFRecordDataset(tf.data.Dataset.list_files(file_pattern))
        dataset = dataset.map(DatasetPipeline.parse_tfrecord_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    @staticmethod
    def count_labels(dataset):
        label_counts = {}
        for _, labels in dataset.unbatch():
            label = labels.numpy()
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1
        return label_counts


# The script below runs only if this file is executed directly
if __name__ == "__main__":
    output_path = './preprocessed_gestures'
    selected_gestures = ['index', 'thumb']  # Replace with your selected gestures
    epochs = 2  # Specify the number of epochs
    pipeline = DatasetPipeline(output_path, selected_gestures=selected_gestures, epochs=epochs)
    train_dataset, val_dataset, test_dataset = pipeline.prepare_dataset()

    print("Dataset is ready for training with augmentation.")
    # Inspect the dataset
    pipeline.inspect_dataset(train_dataset)

    # Save datasets to TFRecord files
    pipeline.save_datasets(train_dataset, val_dataset, test_dataset)

    # Load datasets from TFRecord files for comparison
    train_tfrecord_dataset = DatasetPipeline.load_tfrecord_dataset('datasets/train.tfrecord')
    val_tfrecord_dataset = DatasetPipeline.load_tfrecord_dataset('datasets/val.tfrecord')
    test_tfrecord_dataset = DatasetPipeline.load_tfrecord_dataset('datasets/test.tfrecord')

    # Compare the original and TFRecord datasets
    pipeline.compare_datasets(train_dataset, train_tfrecord_dataset)
    pipeline.compare_datasets(val_dataset, val_tfrecord_dataset)
    pipeline.compare_datasets(test_dataset, test_tfrecord_dataset)

    # Count labels in the datasets
    train_label_counts = pipeline.count_labels(train_tfrecord_dataset)
    val_label_counts = pipeline.count_labels(val_tfrecord_dataset)
    test_label_counts = pipeline.count_labels(test_tfrecord_dataset)

    print(f"Label counts in training dataset: {train_label_counts}")
    print(f"Label counts in validation dataset: {val_label_counts}")
    print(f"Label counts in test dataset: {test_label_counts}")

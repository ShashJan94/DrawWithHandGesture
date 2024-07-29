import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd

class PreVisualization:
    def __init__(self, data_dir, output_dir='figs', num_samples=3):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.num_samples = num_samples
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def save_plot(self, fig, plot_name):
        # Ensure the directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        fig.savefig(os.path.join(self.output_dir, plot_name))
        plt.close(fig)

    def show_sample_images(self, gesture_folder, axes):
        gesture_path = os.path.join(self.data_dir, gesture_folder)
        images = os.listdir(gesture_path)[:self.num_samples]

        for i, img_file in enumerate(images):
            img_path = os.path.join(gesture_path, img_file)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for correct color display
            axes[0, i].imshow(img)
            axes[0, i].axis('off')
            axes[0, i].set_title(f'Sample {i + 1}')
        axes[0, 0].set_ylabel('Sample Images')

    def plot_pixel_intensity_histogram(self, gesture_folder, axes):
        gesture_path = os.path.join(self.data_dir, gesture_folder)
        images = os.listdir(gesture_path)[:self.num_samples]

        for i, img_file in enumerate(images):
            img_path = os.path.join(gesture_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            axes[1, i].hist(img.ravel(), bins=256, color='orange')
            axes[1, i].set_xlabel('Intensity')
            axes[1, i].set_ylabel('Frequency')
            axes[1, i].set_title(f'Pixel Intensity Histogram')
        axes[1, 0].set_ylabel('Intensity Histogram')

    def plot_waveform(self, gesture_folder, axes):
        gesture_path = os.path.join(self.data_dir, gesture_folder)
        images = os.listdir(gesture_path)[:self.num_samples]

        for i, img_file in enumerate(images):
            img_path = os.path.join(gesture_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_waveform = img.ravel()

            axes[2, i].plot(img_waveform[:1000], color='blue')  # Plotting the first 1000 pixel values
            axes[2, i].set_xlabel('Pixel Index')
            axes[2, i].set_ylabel('Intensity')
            axes[2, i].set_title(f'Waveform of Pixel Intensities')
        axes[2, 0].set_ylabel('Waveform')

    def plot_color_channel_histogram(self, gesture_folder, axes):
        gesture_path = os.path.join(self.data_dir, gesture_folder)
        images = os.listdir(gesture_path)[:self.num_samples]

        for i, img_file in enumerate(images):
            img_path = os.path.join(gesture_path, img_file)
            img = cv2.imread(img_path)
            color = ('b', 'g', 'r')

            for j, col in enumerate(color):
                hist = cv2.calcHist([img], [j], None, [256], [0, 256])
                axes[3, i].plot(hist, color=col)

            axes[3, i].set_xlabel('Intensity')
            axes[3, i].set_ylabel('Frequency')
            axes[3, i].set_title(f'Color Channel Histogram')
        axes[3, 0].set_ylabel('Color Histogram')

    def plot_cdf(self, gesture_folder, axes):
        gesture_path = os.path.join(self.data_dir, gesture_folder)
        images = os.listdir(gesture_path)[:self.num_samples]

        for i, img_file in enumerate(images):
            img_path = os.path.join(gesture_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            hist, bins = np.histogram(img.flatten(), 256, [0, 256])
            cdf = hist.cumsum()
            cdf_normalized = cdf * hist.max() / cdf.max()

            axes[4, i].plot(cdf_normalized, color='b')
            axes[4, i].hist(img.flatten(), 256, [0, 256], color='r', alpha=0.5)
            axes[4, i].set_xlabel('Intensity')
            axes[4, i].set_ylabel('Frequency')
            axes[4, i].set_title(f'Cumulative Distribution Function')
        axes[4, 0].set_ylabel('CDF')

    def plot_boxplot(self, gesture_folder, axes):
        gesture_path = os.path.join(self.data_dir, gesture_folder)
        images = os.listdir(gesture_path)[:self.num_samples]

        for i, img_file in enumerate(images):
            img_path = os.path.join(gesture_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # Divide the image into 4 quadrants
            h, w = img.shape
            quadrants = [
                img[:h // 2, :w // 2].flatten(),  # Top-left
                img[:h // 2, w // 2:].flatten(),  # Top-right
                img[h // 2:, :w // 2].flatten(),  # Bottom-left
                img[h // 2:, w // 2:].flatten()  # Bottom-right
            ]

            # Prepare data for box plots
            data = {'Top-left': quadrants[0], 'Top-right': quadrants[1], 'Bottom-left': quadrants[2],
                    'Bottom-right': quadrants[3]}
            df = pd.DataFrame(data)

            sns.boxplot(data=df, orient='h', ax=axes[5, i], palette="Set2", showfliers=False)
            axes[5, i].set_xlabel('Pixel Intensity')
            axes[5, i].set_title(f'Boxplot of Pixel Intensities')
        axes[5, 0].set_ylabel('Boxplot')

    def plot_image_distribution(self):
        gesture_folders = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
        gesture_counts = {}

        for gesture_folder in gesture_folders:
            gesture_path = os.path.join(self.data_dir, gesture_folder)
            num_images = len(os.listdir(gesture_path))
            gesture_counts[gesture_folder] = num_images

        gestures = list(gesture_counts.keys())
        counts = list(gesture_counts.values())

        fig = plt.figure(figsize=(12, 8))
        plt.bar(gestures, counts, color='skyblue')
        plt.xlabel('Gestures')
        plt.ylabel('Number of Images')
        plt.title('Distribution of Images per Gesture')
        plt.xticks(rotation=45)

        self.save_plot(fig, 'image_distribution.png')

    def plot_pca(self):
        gesture_folders = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
        all_images = []
        all_labels = []

        for gesture_folder in gesture_folders:
            gesture_path = os.path.join(self.data_dir, gesture_folder)
            images = os.listdir(gesture_path)[:self.num_samples]

            for img_file in images:
                img_path = os.path.join(gesture_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = img.flatten()
                all_images.append(img)
                all_labels.append(gesture_folder)

        all_images = np.array(all_images)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(all_images)

        fig = plt.figure(figsize=(10, 5))
        for i, label in enumerate(np.unique(all_labels)):
            plt.scatter(pca_result[np.array(all_labels) == label, 0], pca_result[np.array(all_labels) == label, 1],
                        label=label)

        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title('PCA of Image Data')
        plt.legend()

        self.save_plot(fig, 'pca.png')

    def generate_plots(self):
        gesture_folders = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]

        # Generate plots for each gesture folder
        for gesture_folder in gesture_folders:
            fig, axes = plt.subplots(6, self.num_samples, figsize=(15, 20))
            fig.suptitle(f'Visualizations for {gesture_folder}', fontsize=20)

            self.show_sample_images(gesture_folder, axes)
            self.plot_pixel_intensity_histogram(gesture_folder, axes)
            self.plot_waveform(gesture_folder, axes)
            self.plot_color_channel_histogram(gesture_folder, axes)
            self.plot_cdf(gesture_folder, axes)
            self.plot_boxplot(gesture_folder, axes)

            self.save_plot(fig, f'visualizations_{gesture_folder}.png')

        # Generate overall plots
        self.plot_image_distribution()
        self.plot_pca()


# Example usage:
data_dir = './preprocessed_gestures'
previs = PreVisualization(data_dir)

print("Generating plots...")
previs.generate_plots()

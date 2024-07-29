import os
import json
import shutil
import cv2


# Set Kaggle credentials before importing the Kaggle API
def set_kaggle_credentials(kaggle_json_path):
    with open(kaggle_json_path, 'r') as f:
        kaggle_creds = json.load(f)
    os.environ['KAGGLE_USERNAME'] = kaggle_creds['username']
    os.environ['KAGGLE_KEY'] = kaggle_creds['key']


# Path to your kaggle.json file
kaggle_json_path = 'D:/pythonProject/DrawWithHandGesture/kaggle.json'  # Replace with the actual path to your
# kaggle.json file
set_kaggle_credentials(kaggle_json_path)

from kaggle.api.kaggle_api_extended import KaggleApi


class Preprocessing:
    def __init__(self, kaggle_dataset, dataset_dir, output_dir, target_shape=(64, 64), num_images=2000):
        self.kaggle_dataset = kaggle_dataset
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.target_shape = target_shape
        self.num_images = num_images
        self.api = self.custom_kaggle_authenticate()
        self.subjects_dir = None

    @staticmethod
    def custom_kaggle_authenticate():
        print("Authenticating with Kaggle...")
        api = KaggleApi()
        api.authenticate()
        print("Kaggle authentication successful.")
        return api

    def clean_up_existing_data(self):
        if os.path.exists(self.dataset_dir):
            print(f"Removing existing dataset directory: {self.dataset_dir}")
            shutil.rmtree(self.dataset_dir)
        print("Cleaned up existing data.")

    def remove_nested_leapgestrecog(self):
        nested_dir = os.path.join(self.dataset_dir, 'leapGestRecog', 'leapGestRecog')
        if os.path.exists(nested_dir):
            print(f"Removing nested directory: {nested_dir}")
            shutil.rmtree(nested_dir)
        print("Nested directory removed.")

    def find_leapgestrecog_dir(self):
        for root, dirs, files in os.walk(self.dataset_dir):
            for dir_name in dirs:
                if dir_name == 'leapGestRecog':
                    return os.path.join(root, dir_name)
        return None

    def download_and_extract(self):
        self.clean_up_existing_data()

        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)

        print("Downloading dataset from Kaggle...")
        self.api.dataset_download_files(self.kaggle_dataset, path=self.dataset_dir, unzip=True)
        print("Dataset downloaded successfully.")
        # Handle nested zip structure
        self.remove_nested_leapgestrecog()
        print("Handled nested zip structure.")
        # Find the correct leapGestRecog directory
        self.subjects_dir = self.find_leapgestrecog_dir()
        if self.subjects_dir:
            print(f"Found leapGestRecog directory: {self.subjects_dir}")
        else:
            print("leapGestRecog directory not found.")

        # Remove the zip file if it exists
        zip_file_path = os.path.join(self.dataset_dir, self.kaggle_dataset.split('/')[-1] + '.zip')
        if os.path.exists(zip_file_path):
            print(f"Removing zip file: {zip_file_path}")
            os.remove(zip_file_path)
        print("Zip file removed if it existed.")

    def preprocess_images(self):
        self.subjects_dir = r"D:/pythonProject/DrawWithHandGesture/kaggle_leapgestrecog/leapGestRecog"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        print("Starting image preprocessing...")

        # Extract gesture folder names from the first subject (00)
        subject_00_path = os.path.join(self.subjects_dir, '00')
        gesture_folders = [d for d in os.listdir(subject_00_path) if os.path.isdir(os.path.join(subject_00_path, d))]
        print(f"Gesture folders: {gesture_folders}")

        # Create directories for each gesture
        for gesture_folder in gesture_folders:
            gesture_name = '_'.join(gesture_folder.split('_')[1:])  # Combine all parts after the first underscore
            gesture_output_path = os.path.join(self.output_dir,
                                               gesture_name)  # Ensure it's within the preprocessed images folder
            os.makedirs(gesture_output_path, exist_ok=True)
            print(f"Created directory for gesture: {gesture_name}")

        # Process each subject and move images to corresponding gesture folders
        for subject_folder in os.listdir(self.subjects_dir):
            subject_path = os.path.join(self.subjects_dir, subject_folder)
            if os.path.isdir(subject_path):
                print(f"Processing subject: {subject_folder}")
                for gesture_folder in os.listdir(subject_path):
                    gesture_path = os.path.join(subject_path, gesture_folder)
                    if os.path.isdir(gesture_path):
                        gesture_name = '_'.join(gesture_folder.split('_')[1:])
                        gesture_output_path = os.path.join(self.output_dir, gesture_name)
                        if not os.path.exists(gesture_output_path):
                            os.makedirs(gesture_output_path)
                        print(f"Processing gesture: {gesture_folder}, output path: {gesture_output_path}")

                        for img_file in os.listdir(gesture_path):
                            img_path = os.path.join(gesture_path, img_file)
                            shutil.copy(img_path, gesture_output_path)
                print(f"Finished processing subject: {subject_folder}")

        print("Image preprocessing completed successfully.")
        self.normalize_images()

    def normalize_images(self):
        print("Starting image normalization...")
        for gesture_folder in os.listdir(self.output_dir):
            gesture_path = os.path.join(self.output_dir, gesture_folder)
            for img_file in os.listdir(gesture_path):
                img_path = os.path.join(gesture_path, img_file)
                img = cv2.imread(img_path)
                img = cv2.resize(img, self.target_shape)
                img = img / 255.0  # Normalize pixel values
                cv2.imwrite(img_path, img * 255)
        print("Image normalization completed successfully.")


# Example usage:
kaggle_dataset = 'gti-upm/leapgestrecog'
dataset_dir = './kaggle_leapgestrecog'
output_dir = './preprocessed_gestures'

preprocessing = Preprocessing(kaggle_dataset, dataset_dir, output_dir, num_images=2000)
#print("Initiating download and extraction...")
#preprocessing.download_and_extract()
print("Initiating preprocessing of images...")
preprocessing.preprocess_images()

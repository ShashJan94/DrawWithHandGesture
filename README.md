
---

# Hand Gesture Drawing App

This project is a hand gesture recognition-based drawing application built using TensorFlow, OpenCV, MediaPipe, and Tkinter. The application captures hand gestures via a webcam and uses a trained neural network model to recognize gestures and perform drawing or erasing actions on a canvas.

## Features

- Real-time hand tracking using MediaPipe
- Gesture recognition using a TensorFlow model
- Drawing and erasing on a canvas based on recognized gestures
- Separate panels for webcam feed and drawing canvas

## Requirements

- Python 3.6+
- TensorFlow 2.0+
- OpenCV
- MediaPipe
- Tkinter (usually included with Python)

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/hand-gesture-drawing-app.git
   cd hand-gesture-drawing-app
   ```

2. Create a virtual environment and activate it:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

1. Ensure you have a trained TensorFlow model saved as `gesture_recognition_model.h5` and place it in the `model` directory. If you need to train the model, follow the instructions in the `Training the Model` section below.

2. Run the application:
   ```sh
   python drawapp.py
   ```

3. Use your webcam to show gestures:
   - **Index Finger**: Start drawing on the canvas.
   - **Thumb**: Start erasing on the canvas.

## Training the Model

If you don't have a pre-trained model, you can train your own using the provided script. 

1. Prepare your dataset with labeled images of gestures (e.g., index finger and thumb) and save them in the `preprocessed_gestures` directory.

2. Run the dataset pipeline to create TFRecord files:
   ```sh
   python datapipeline.py
   ```

3. Train the model using the generated TFRecord files:
   ```sh
   python gesturerecognitionmodel.py
   ```

## Problems and Limitations

### Static Image Training
- The model was trained on static grayscale images of hand gestures, which poses several challenges:
  - **Lack of Temporal Data**: The model does not account for the motion and context that a sequence of images provides, making it less effective in recognizing gestures in real-time video feeds.
  - **Static Backgrounds**: Training on static images often involves less variation in background and lighting conditions compared to real-world scenarios.

### Gesture Recognition Issues
- The model may have difficulty distinguishing between gestures, particularly when switching from one gesture to another, due to the static nature of the training data.
- Inconsistent lighting, varying hand positions, and different backgrounds can affect the model's performance.

### Performance
- Real-time prediction with the model can be slow, affecting the responsiveness of the application. This is partly due to the model processing each frame individually without leveraging sequential data.

## Project Structure

```
hand-gesture-drawing-app/
│
├── model/
│   ├── gesture_recognition_model.h5      # Pre-trained TensorFlow model
│
├── datasets/
│   ├── train.tfrecord                    # Training dataset in TFRecord format
│   ├── val.tfrecord                      # Validation dataset in TFRecord format
│   ├── test.tfrecord                     # Test dataset in TFRecord format
│
├── preprocessed_gestures/                # Directory for preprocessed gesture images
│
├── drawapp.py                            # Main application script
├── datapipeline.py                       # Script for preparing the dataset
├── gesturerecognitionmodel.py            # Script for training the model
├── requirements.txt                      # List of required packages
├── README.md                             # Project README file
```

## Notes

- Ensure your webcam is properly connected and accessible by OpenCV.
- The application may require some fine-tuning depending on the quality and characteristics of your dataset.
- The model provided in the repository is trained on static grayscale images of gestures. Adjustments may be necessary for better performance on video inputs.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

- [TensorFlow](https://www.tensorflow.org/)
- [OpenCV](https://opencv.org/)
- [MediaPipe](https://mediapipe.dev/)
- [Tkinter](https://wiki.python.org/moin/TkInter)

---

Feel free to customize this README file to better fit your specific needs and project details.
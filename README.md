# Tumor Classification and Localization with Deep Learning

This project implements a deep learning-based system for detecting and classifying tumors from medical images. It uses Convolutional Neural Networks (CNN) for tumor classification and segmentation. The system also includes tumor localization features, leveraging segmentation to detect tumor regions and estimate their characteristics.

## Features

- Tumor classification (Malignant, Benign, or Normal)
- Tumor localization and segmentation
- Model training with ResNet-101 as base architecture
- Tumor region extraction using contour detection
- Training and validation of models with accuracy and loss tracking
- Visualization of model performance with graphs

## Requirements

- Python 3.x
- Django
- Keras
- TensorFlow
- OpenCV
- NumPy
- scikit-learn
- Matplotlib


Setup Instructions
1. Clone the repository:
```
git clone https://github.com/pankaj7322/Brain-Tumour-Detection.git
cd Brain-Tumor-Detection
```
2. Set up a Virtual Ennvironment:
```
python -m venv venv
venv/bin/activate
```
3. Install dependencies:
```
pip install -r requirements.txt
```
4. Run the Django Development server: 
```
python manage.py migrate
```
6. Navigate to http://127.0.0.1:8000/ in your web browser to access the app.

## Overview of the Code

1. Login and Authentication

The login_view and logout_view handle user authentication. Users can log in with their credentials, and upon successful authentication, they are redirected to the home page.
2. Model Loading

    Pre-trained Segmented Model: The system loads a pre-trained tumor segmentation model (stored in JSON and H5 format) used for detecting tumor regions from medical images.
    CNN Model: The system can train a custom CNN model using the ResNet-101 architecture for classification purposes. The model is trained on a dataset and saved for later use.

3. Dataset Upload and Preprocessing

    Users can upload image datasets, which are stored and processed to create training and validation data. The images are resized and converted to grayscale for classification and segmentation tasks.
    The dataset is divided into two classes: Normal (no tumor) and Malignant (tumor present).

4. Tumor Classification

    When a user uploads an image, the system first checks the image classification using the CNN model (whether it contains a benign or malignant tumor).
    If a tumor is detected, the system localizes the tumor region and applies contour detection to visualize the tumor's position in the image.

5. Tumor Localization

    The system uses a segmentation model to create a binary mask of the tumor region.
    Contours are then detected in the segmented image, and the largest contour (assumed to be the tumor) is localized by drawing a circle around it.

6. Training the Model

    The model is trained using a ResNet-101 architecture as the base model, with custom layers added for classification.
    The model's performance (accuracy and loss) is tracked and stored in a pickle file, which can be used to plot the training progress.

7. Visualization of Training Progress

    The graph function generates a plot of the model's training accuracy and loss over epochs and saves it as an image file.
    This plot is displayed on the web interface for visual feedback.
    
    
## file stucture

```
├── media/                    # Uploaded files and localized images
├── Model/                    # Pre-trained models and weights
│   ├── segmented_model.json  # Model architecture for segmentation
│   ├── segmented_weights.h5 # Weights for the segmentation model
│   ├── model.json            # Model architecture for CNN classifier
│   ├── model_weights.h5      # Weights for the CNN classifier
│   ├── history.pckl          # Training history (accuracy, loss)
├── static/                   # Generated static files like graphs
├── tumor_classification/     # Django app containing the main logic
│   ├── templates/            # HTML templates
│   ├── views.py              # Main views and logic
├── manage.py                 # Django management script
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```
## Contributing

Feel free to open issues, submit pull requests, or suggest improvements. Make sure to follow the standard GitHub workflow:

    1. Fork the repository.
    2. Create a new branch for your changes.
    3. Submit a pull request with a clear description of your changes.

## License

This project is licensed the MIT License


import os
import cv2
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from keras.models import Sequential, model_from_json
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pickle
import os
from django.conf import settings



from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login

def login_view(request):
    error = None
    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)  # Pass both request and user here
            return redirect('home')  # Redirect to a home page or dashboard
        else:
            error = "Invalid username or password."

    return render(request, 'login.html', {'error': error})



from django.contrib.auth import logout

def logout_view(request):
    logout(request)
    return redirect('login')







# Global variables
X = []
Y = []
accuracy = 0
#classifier = None
segmented_model = None
disease = ['Normal', 'Malignant']
DATASET_DIR = "dataset"  # Define your dataset directory
with open('Model/segmented_model.json', "r") as json_file:
    loaded_model_json = json_file.read()
    segmented_model = model_from_json(loaded_model_json)
json_file.close()    
segmented_model.load_weights("Model/segmented_weights.h5")
#segmented_model._make_predict_function()

# Home page view
def home(request):
    return render(request, 'home.html')


# Upload dataset
def upload_dataset(request):
    if request.method == 'POST' and request.FILES.get('file'):
        myfile = request.FILES['file']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        return render(request, 'home.html', {'message': "Data uploaded successfully"})
    return render(request, 'home.html', {'message': "No file uploaded"})


# Dataset Preprocessing
def dataset_preprocessing(request):
    global X, Y
    X, Y = [], []
    data_path = os.path.join(DATASET_DIR)

    if os.path.exists('Model/myimg_data.txt.npy') and os.path.exists('Model/myimg_label.txt.npy'):
        X = np.load('Model/myimg_data.txt.npy')
        Y = np.load('Model/myimg_label.txt.npy')
    else:
        for label, folder in enumerate(['no', 'yes']):
            folder_path = os.path.join(data_path, folder)
            if not os.path.exists(folder_path):
                return render(request, 'home.html', {'message': f"Folder '{folder}' not found in dataset path."})
            for file in os.listdir(folder_path):
                try:
                    img = cv2.imread(os.path.join(folder_path, file), cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (64, 64))
                    X.append(img.reshape(64, 64, 1))
                    Y.append(label)
                except Exception as e:
                    return render(request, 'home.html', {'message': f"Error processing file {file}: {str(e)}"})

        X, Y = np.array(X), np.array(Y)
        np.save("Model/myimg_data.txt", X)
        np.save("Model/myimg_label.txt", Y)

    response_data = {
        "total_images": len(X),
        "total_classes": len(set(Y)),
        "class_labels": disease
    }
    return render(request, 'home.html', {'message': f"Preprocessing completed: {response_data}"})


# Load Pre-trained Model
def load_segmented_model():
    global segmented_model
    try:
        with open('Model/segmented_model.json', "r") as json_file:
            segmented_model = model_from_json(json_file.read())
        segmented_model.load_weights("Model/segmented_weights.h5")
    except Exception as e:
        print(f"Error loading segmented model: {str(e)}")

load_segmented_model()


# Train CNN Model
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, GlobalAveragePooling2D

# Train CNN Model
def train_model(request):
    global classifier, accuracy, X, Y
    if len(X) == 0 or len(Y) == 0:
        return render(request, 'home.html', {'message': "Dataset not loaded or empty. Please preprocess the dataset first."})

    YY = to_categorical(Y)
    x_train, x_test, y_train, y_test = train_test_split(X, YY, test_size=0.2, random_state=0)

    if os.path.exists('Model/model.json'):
        with open('Model/model.json', "r") as json_file:
            classifier = model_from_json(json_file.read())
        classifier.load_weights("Model/model_weights.h5")
    else:
        # Define ResNet-101 base model
        input_shape = (64, 64, 3)  # ResNet requires 3 channels; update if needed
        resnet_base = ResNet101(weights='imagenet', include_top=False, input_tensor=Input(shape=input_shape))

        # Freeze base model layers
        for layer in resnet_base.layers:
            layer.trainable = False

        # Add custom classification layers
        x = resnet_base.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        predictions = Dense(2, activation='softmax')(x)

        # Create model
        classifier = Model(inputs=resnet_base.input, outputs=predictions)
        classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train model
        hist = classifier.fit(x_train, y_train, batch_size=16, epochs=10, validation_data=(x_test, y_test), shuffle=True, verbose=2)

        # Save model
        classifier.save_weights('Model/model_weights.h5')
        with open("Model/model.json", "w") as json_file:
            json_file.write(classifier.to_json())
        with open('Model/history.pckl', 'wb') as f:
            pickle.dump(hist.history, f)

    # Load and calculate accuracy
    with open('Model/history.pckl', 'rb') as f:
        data = pickle.load(f)
    accuracy = data['accuracy'][-1] * 100

    return render(request, 'home.html', {'message': f"Model trained successfully with accuracy: {accuracy:.2f}%"})

def cropTumorRegion(orig, thresh):
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    min_area = 0.95 * 180 * 35
    max_area = 1.05 * 180 * 35
    result = orig.copy()
    life = 0
    for c in contours:
        area = cv2.contourArea(c)
        if life == 0:
            life = len(c)
        cv2.drawContours(result, [c], -1, (255, 0, 0), 10)
        if area > min_area and area < max_area:
            cv2.drawContours(result, [c], -1, (255, 0, 0), 10)
    return result, life

def getTumorRegion(filename):
    global segmented_model

    # Read the original image (color) for tumor segmentation and localization
    orig = cv2.imread(filename)  # Load the original image (color)
    orig_resized = orig.copy()  # Keep a copy of the original for processing

    # Preprocess for segmentation model (resize only for model input)
    img = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for segmentation
    img_resized = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
    img_resized = img_resized.reshape(1, 64, 64, 1)  # Reshape for model input
    img_resized = (img_resized - 127.0) / 127.0  # Normalize if needed

    # Predict tumor segmentation using the model
    preds = segmented_model.predict(img_resized)
    preds = preds[0]  # Get the output for the image

    # Resize the predicted mask to the original size for visualization
    segmented_image = cv2.resize(preds, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_CUBIC)

    # Convert the segmented image to uint8 for contour detection
    segmented_image = np.uint8(segmented_image * 255)

    # Perform thresholding to ensure binary image
    _, thresholded_image = cv2.threshold(segmented_image, 127, 255, cv2.THRESH_BINARY)

    # Perform edge detection and estimate lifespan
    edge_detection, lifespan = cropTumorRegion(orig, thresholded_image)  # Pass original image size here

    # Return the segmented image, edge detection, and lifespan
    return thresholded_image, edge_detection, lifespan

def tumor_classification(request):
    if request.method == 'POST' and request.FILES.get('image-name'):
        myfile = request.FILES['image-name']
        fs = FileSystemStorage(location='media/')  # Save uploaded files to media/
        filename = fs.save(myfile.name, myfile)
        file_path = fs.path(filename)  # Full path of the uploaded image

        try:
            # Preprocess the image for classification model
            img = cv2.imread(file_path)
            img_resized = cv2.resize(img, (64, 64))  # Resize to the expected input size for classifier
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            img_resized = np.expand_dims(img_gray, axis=-1)  # Add channel dimension for grayscale

            im2arr = np.array(img_resized).reshape(1, 64, 64, 1)  # Reshape for the model input
            XX = np.asarray(im2arr) / 255.0  # Normalize the input if model was trained with normalized data

            # Predict the classification
            predicts = classifier.predict(XX)
            cls = np.argmax(predicts)
            
            # Default classification message
            result_message = ""

            localized_tumor_path = None  # Default to None if no tumor detected

            if cls == 1:  # If tumor is detected (Benign or Malignant)
                # Use segmentation model to find tumor location
                segmented_image, edge_image, lifespan = getTumorRegion(file_path)

                # Log segmented image and check if contours are detected
                print("Segmented Image Shape:", segmented_image.shape)

                # Assuming segmented_image provides the binary mask of the tumor
                contours, _ = cv2.findContours(segmented_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    # Log the number of contours detected
                    print(f"Number of contours detected: {len(contours)}")
                    
                    # Get the largest contour (assuming it represents the tumor)
                    largest_contour = max(contours, key=cv2.contourArea)

                    # Log the area of the largest contour
                    print(f"Largest contour area: {cv2.contourArea(largest_contour)}")

                    if cv2.contourArea(largest_contour) > 100:  # Threshold to avoid small contours
                        # Get the minimum enclosing circle for the largest contour
                        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
                        tumor_center = (int(x), int(y))
                        tumor_radius = int(radius)  # Use the radius without shrinking

                        # Log the circle center and radius
                        print(f"Tumor center: {tumor_center}, Tumor radius: {tumor_radius}")

                        # Convert the original grayscale image to BGR for drawing colored circle
                        img_with_tumor = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Retain original color dimensions
                        cv2.circle(img_with_tumor, tumor_center, tumor_radius, (255, 0, 0), 2)  # Red circle (BGR)

                        # Save localized image, maintaining the original size
                        localized_tumor_path = os.path.join(f'tumor_localized_{myfile.name}')
                        localized_tumor_full_path = os.path.join(settings.MEDIA_ROOT, f'tumor_localized_{myfile.name}')
                        cv2.imwrite(localized_tumor_full_path, img_with_tumor)

                        result_message = f"Classification Result: {disease[cls]} | Predicted Lifespan: {lifespan} months"
                    else:
                        result_message = "Classification Result: Tumor contour too small to localize"
                else:
                    result_message = "Classification Result: Normal (No Tumor Detected)"
            else:
                result_message = "Classification Result: Normal (No Tumor Detected)"

            print(f"Localized tumor image URL: {fs.url(localized_tumor_path) if localized_tumor_path else None}")

            # Return result and file URLs to the template
            return render(request, 'home.html', {
                'message': result_message,
                'uploaded_file_url': fs.url(filename),  # URL of the uploaded file
                'tumor_localized_url': fs.url(localized_tumor_path) if localized_tumor_path else None
            })

        except Exception as e:
            return render(request, 'home.html', {'message': f"Error in prediction: {str(e)}"})

    return render(request, 'home.html', {'message': "No image uploaded or invalid request."})


import matplotlib
matplotlib.use('Agg')

def graph(request):
    try:
        # Load the history from the pickle file
        with open('Model/history.pckl', 'rb') as f:
            data = pickle.load(f)

        # Ensure that 'loss' and 'accuracy' are lists (or arrays)
        loss = data.get('loss', [])
        accuracy = data.get('accuracy', [])

        # Convert loss and accuracy to lists if they aren't already
        if not isinstance(loss, list):
            loss = list(loss)
        if not isinstance(accuracy, list):
            accuracy = list(accuracy)

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.grid(True)
        plt.xlabel('Training Epoch')
        plt.ylabel('Accuracy/Loss')
        plt.plot(loss, color='red', label='Loss')
        plt.plot(accuracy, color='green', label='Accuracy')
        plt.legend(loc='upper left')
        plt.title('Model Training Accuracy & Loss')



        # Ensure the directory exists
        static_dir = os.path.join(settings.BASE_DIR, 'static')
        if not os.path.exists(static_dir):
          os.makedirs(static_dir) 

        # Save the plot to the static folder
        graph_path = os.path.join(settings.BASE_DIR, 'static', 'training_graph.png')
        plt.savefig(graph_path)

        # Return the graph in the context to the template
        return render(request, 'graph.html', {'graph': 'training_graph.png'})

    except Exception as e:
        # Handle errors gracefully
        return render(request, 'home.html', {'message': f"Error displaying graph: {str(e)}"})
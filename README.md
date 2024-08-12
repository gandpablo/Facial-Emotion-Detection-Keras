This project focuses on creating an emotion classification model from images using Python. The dataset I used is from Kaggle: https://www.kaggle.com/datasets/msambare/fer2013

The main goal is to put this model to practical use by developing an app with Tkinter (also in Python). The app uses a pre-trained model from the OpenCV library to detect faces in images, which are then passed to the model I trained to recognize emotions.

The app continuously captures images from the camera in real-time, allowing it to constantly detect the emotion of the person in front of the camera.

In this repository, you’ll find a few key files:

tensores.ipynb: A Jupyter notebook that generates an .npy file containing all images converted into numpy arrays X for the images and Y for the emotion labels, encoded in one hot.
entrenamiento.ipynb: Another notebook where I trained some models, mostly convolutional neural networks (CNNs).
Model and Validation Data: A ZIP file containing the trained Keras final model and a subset of the data (X and Y) for validation. If you need the full dataset in npy format, you can run the code from tensores.ipynb.
Application: The app that opens the camera, detects faces using an OpenCV model, and predicts the emotion with the model I trained.
probar_modelo.ipynb: A notebook for testing and validating the model.

The model I trained has an accuracy of only 58%, but it works surprisingly well in real-world scenarios.

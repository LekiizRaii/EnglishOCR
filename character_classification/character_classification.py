import pickle
# load model from pickle file
with open("iris_classifier_model.pkl", 'rb') as file:  
    model = pickle.load(file)
import cv2
import matplotlib.pyplot as plt
import numpy as np
# Load the unknown image
unknown_image = cv2.imread('testimg.png')  # Replace 'unknown_image.jpg' with the path to your image
unknown_image = cv2.cvtColor(unknown_image, cv2.COLOR_BGR2GRAY)
# Resize the image to (40,40)
unknown_image = cv2.resize(unknown_image, (40,40))

# Normalize pixel values to the range [0, 1]
unknown_image = unknown_image/ 255.0
print(unknown_image.shape)
plt.imshow(unknown_image)
# Make predictions
predictions = model.predict(np.expand_dims(unknown_image, axis=0))
# Interpret the predictions
predicted_class_index = np.argmax(predictions)
factlabel=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
       'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
       'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f',
       'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
       'u', 'v', 'w', 'x', 'y', 'z']
print(factlabel[predicted_class_index])
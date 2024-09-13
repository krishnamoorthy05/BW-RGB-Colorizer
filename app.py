
# Import the necessary packages
import numpy as np
import cv2
import streamlit as st
from PIL import Image

# Define the colorizer function
def colorizer(img):
    # Convert the image to grayscale and back to RGB
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

    # Load the pre-trained model and cluster center points from disk
    prototxt =r"/Users/krishna/Desktop/Colorizer/models_colorization_deploy_v2.prototxt"
    model = r"/Users/krishna/Desktop/Colorizer/colorization_release_v2-2.caffemodel"
    points = r"/Users/krishna/Desktop/Colorizer/pts_in_hull.npy"
   
    # Load the model and points using OpenCV's DNN module
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    pts = np.load(points)

    # Add cluster centers as 1x1 convolutions to the model
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    # Convert image to the LAB color space
    scaled = img_rgb.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)

    # Resize the LAB image to 224x224 and extract the L channel
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50  # Mean-centering

    # Predict the 'a' and 'b' channels
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (img_rgb.shape[1], img_rgb.shape[0]))

    # Combine the original L channel with predicted 'ab' channels
    L_original = cv2.split(lab)[0]
    colorized = np.concatenate((L_original[:, :, np.newaxis], ab), axis=2)

    # Convert the LAB image back to RGB
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
    colorized = np.clip(colorized, 0, 1)  # Ensure all values are within [0, 1] range
    colorized = (255 * colorized).astype("uint8")  # Convert to 8-bit representation

    return colorized

##########################################################################################################

# Streamlit user interface
st.write("# Colorize Your Black and White Image")
st.write("This app allows you to colorize your B&W images.")

# File uploader
file = st.sidebar.file_uploader("Please upload an image file (JPG or PNG format)", type=["jpg", "png"])

if file is None:
    st.text("You haven't uploaded an image file yet.")
else:
    # Open and process the uploaded image
    image = Image.open(file)
    img = np.array(image)

    # Display original image
    st.text("Original Image:")
    st.image(image, use_column_width=True)

    # Colorize the image and display it
    st.text("Colorized Image:")
    colorized_img = colorizer(img)
    st.image(colorized_img, use_column_width=True)

    print("Done!")  # Output confirmation in the console (optional for debugging)

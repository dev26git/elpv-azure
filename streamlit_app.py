import streamlit as st
from streamlit_image_select import image_select
from PIL import Image
import numpy as np
import cv2
import os
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('resources/models/trained_cnn.keras')
SAMPLE_IMAGES_FOLDER = "image_repo/Panel_Images"


# Utility functions

def extract_cells_from_panel(panel_image, cell_width, cell_height):

    # Get the dimensions of the solar panel image
    panel_height, panel_width = panel_image.shape

    # Calculate the number of rows and columns based on the cell dimensions
    num_rows = panel_height // cell_height
    num_cols = panel_width // cell_width

    # Initialize an empty list to store individual cell images
    cell_images = []

    # Iterate over the grid to extract individual cell images
    for i in range(num_rows):
        for j in range(num_cols):
            # Calculate the coordinates for slicing
            y_start = i * cell_height
            y_end = (i + 1) * cell_height
            x_start = j * cell_width
            x_end = (j + 1) * cell_width

            # Slice the panel image to extract the cell
            cell = panel_image[y_start:y_end, x_start:x_end]

            # Append the extracted cell to the list
            cell_images.append(cell)

    # Convert the list of cell images to a numpy array
    cell_images_array = np.array(cell_images)

    return cell_images_array


def image_preprocessing(img):
    # equalize
    img = img.astype('uint8')
    clahe = cv2.createCLAHE(tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = np.expand_dims(img, 2)
    
    return img


# Function to highlight cells based on class labels
def highlight_cells(solar_panel, class_labels):
    # Define colors for each class (BGR format)
    colors = [(0, 255, 0), (255, 165, 0), (255, 0, 0)]  # Green, Orange, Red

    # Get the height and width of each cell
    cell_height = 300
    cell_width = 300

    # Convert grayscale image to 3-channel BGR image
    solar_panel = cv2.cvtColor(solar_panel, cv2.COLOR_GRAY2BGR)
    
    # Create a mask for the current cell
    mask = np.zeros_like(solar_panel)

    # Highlight cells based on class labels
    for i in range(6):
        for j in range(10):
            cell_index = i * 10 + j
            class_label = class_labels[cell_index]
            color = colors[class_label]
            
            # Define the region of interest (ROI) for the current cell
            y_start = i * cell_height
            y_end = (i + 1) * cell_height
            x_start = j * cell_width
            x_end = (j + 1) * cell_width
            
            mask[y_start:y_end, x_start:x_end] = color

    # Apply alpha blending to overlay the mask on the solar panel
    alpha = 0.25  # Adjust the alpha value as needed
    solar_panel = cv2.addWeighted(solar_panel, 1 - alpha, mask, alpha, 0)

    return solar_panel






# Main function
def main():
    st.set_page_config(page_title='Solar Panel Fault Detection', layout="wide")

    # Symbiosis Logo
    with st.container():
        _, middle, _ = st.columns((5, 1, 5))
        with middle:
            st.image("image_repo/Logo_of_Symbiosis_International_University.svg.png")
    with st.container():
        _, middle, _ = st.columns((4, 8, 1))
        with middle:
            st.subheader("Symbiosis Institute of Technology")

    st.write("##")
    st.write("##")
    st.write("##")

    # HEADER SECTION
    st.title(":blue[Fault Detection in Solar Panels] :camera:")

    st.write("##")
    st.subheader('An automated Computer Vision and Deep Learning approach to detect defective solar cells.')
    st.write("It uses image processing and deep learning to find potentially faulty PV cells.")
    st.write("A Convolutional Neural Network is used for classification.")

    st.divider()

    st.subheader("Framework")
    with st.container():
        _, middle, _ = st.columns((1, 8, 1))
        with middle:
            st.image("image_repo/Pipeline.png")

    st.divider()

    st.write("##")
    st.write("##")

    with st.container():
        panel_img = image_select(
            label="Select an image",
            images=[
            cv2.imread(os.path.join(SAMPLE_IMAGES_FOLDER, filename), cv2.IMREAD_GRAYSCALE)
            for filename in os.listdir(SAMPLE_IMAGES_FOLDER)
            if filename.endswith(".jpg")
        ]
        # captions=actual_volumes,
        # return_value='index'
    )

    st.write("##")
    st.write("##")

    if panel_img is not None:
        st.subheader("Fault Detection Module Output")
        st.info("Please wait for the model to work its magic.")

        # Sliding Window Algorithm to extract individual cells
        cell_imgs = extract_cells_from_panel(panel_img, 300, 300)

        # Apply the equalize function to each image in 'images'
        enhanced_images = np.array([image_preprocessing(img) for img in cell_imgs])

        # Reshape images to 224,224 to match model input
        reshaped_images = np.zeros((60, 224, 224, 1), dtype=np.uint8)
        for i in range(60):
            reshaped_images[i, :, :, 0] = cv2.resize(enhanced_images[i], (224, 224))

        # Make defect predictions for each cell
        predictions = model.predict(reshaped_images)

        # find the class with the maximum probability
        predictions = np.argmax(predictions, axis=1)

        # Count number of images in each class
        class_counts = [0,0,0]
        for i in predictions:
            class_counts[i]+=1

        panel_img_highlighted = highlight_cells(panel_img, predictions)

        _, middle, _ = st.columns((1, 8, 1))
        with middle:
            st.image(panel_img_highlighted)
        
        # Define your data
        labels = ['Normal', 'Potentially-Defective', 'Defective']

        st.subheader("Distribution of Faults Detected")

        # List the distribution of number of images of each class
        st.write(f"**Normal: {class_counts[0]}**")
        st.write(f"**Potentially-Defective: {class_counts[1]}**")
        st.write(f"**Defective: {class_counts[2]}**")

if __name__ == "__main__":
    main()

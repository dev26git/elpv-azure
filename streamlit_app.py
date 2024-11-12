import streamlit as st
from PIL import Image
import numpy as np
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
    # Normalize the image to the range [0, 1]
    img = np.array(img, dtype=np.float32) / 255.0

    # Expand dimensions to match model input (224, 224, 1)
    img = np.expand_dims(img, axis=-1)
    
    return img


# Function to display distribution of fault types
def display_fault_distribution(predictions):
    # Count number of images in each class
    class_counts = [0, 0, 0]
    for i in predictions:
        class_counts[i] += 1

    st.subheader("Distribution of Faults Detected")
    st.write(f"**Normal: {class_counts[0]}**")
    st.write(f"**Potentially-Defective: {class_counts[1]}**")
    st.write(f"**Defective: {class_counts[2]}**")


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

    # Select image from dropdown
    panel_img_filename = st.selectbox(
        "Select a Solar Panel Image",
        [filename for filename in os.listdir(SAMPLE_IMAGES_FOLDER) if filename.endswith(".jpg")]
    )

    panel_img_path = os.path.join(SAMPLE_IMAGES_FOLDER, panel_img_filename)
    panel_img = Image.open(panel_img_path).convert('L')  # Convert to grayscale

    st.write("##")
    st.write("##")

    if panel_img is not None:
        st.subheader("Fault Detection Module Output")

        # Sliding Window Algorithm to extract individual cells
        cell_imgs = extract_cells_from_panel(np.array(panel_img), 300, 300)

        # Apply preprocessing to each image in 'cell_imgs'
        enhanced_images = np.array([image_preprocessing(Image.fromarray(img)) for img in cell_imgs])

        # Reshape images to (224, 224, 1) to match model input
        reshaped_images = np.zeros((60, 224, 224, 1), dtype=np.float32)
        for i in range(60):
            # Ensure correct format by converting to uint8 before resizing
            pil_img = Image.fromarray(enhanced_images[i].astype(np.uint8).squeeze())  # Convert to uint8
            reshaped_images[i, :, :, 0] = np.array(pil_img.resize((224, 224)), dtype=np.float32) / 255.0

        # Make defect predictions for each cell
        predictions = model.predict(reshaped_images)

        # Find the class with the maximum probability
        predictions = np.argmax(predictions, axis=1)

        st.write(f"Processed {panel_img_filename}")

        # Display the fault distribution
        display_fault_distribution(predictions)


if __name__ == "__main__":
    main()
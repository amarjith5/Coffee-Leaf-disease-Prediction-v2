import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import os
from PIL import Image
import io
import matplotlib.pyplot as plt
import logging
from utils import preprocess_image, segment_leaf, apply_clahe, extract_features

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Coffee Leaf Disease Detection",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define paths to dataset
DATASET_PATH = "data"
TRAIN_IMAGES_PATH = os.path.join(DATASET_PATH, "train", "images")
TRAIN_MASKS_PATH = os.path.join(DATASET_PATH, "train", "masks")
TEST_IMAGES_PATH = os.path.join(DATASET_PATH, "test", "images")
TEST_MASKS_PATH = os.path.join(DATASET_PATH, "test", "masks")
EXAMPLE_IMAGES_PATH = os.path.join(DATASET_PATH, "examples") if os.path.exists(os.path.join(DATASET_PATH, "examples")) else None

# Define class names
CLASS_NAMES = ["Coffee Leaf Rust", "Coffee Berry Disease", "Cercospora Leaf Spot", "Healthy"]

# Load the pre-trained model
@st.cache_resource
def load_disease_model():
    try:
        model_path = 'models/coffee_disease_model.h5'
        if not os.path.exists(model_path):
            st.error(f"Model file not found at {model_path}. Please train the model first.")
            return None
        
        model = load_model(model_path)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        logger.error(f"Failed to load model: {e}")
        return None

# Function to make predictions
def predict_disease(img, model):
    try:
        img_array = preprocess_image(img)
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        return predicted_class, confidence, predictions[0]
    except Exception as e:
        logger.error(f"Error in disease prediction: {e}")
        st.error(f"Failed to predict disease: {str(e)}")
        return 0, 0.0, np.zeros(len(CLASS_NAMES))

# Function to display disease information
def display_disease_info(disease_name):
    disease_info = {
        "Coffee Leaf Rust": {
            "causative_agent": "Hemileia vastatrix (fungus)",
            "symptoms": "Orange-yellow powder on the underside of leaves; yellow spots on the upper surface",
            "effects": "Leaf drop, reduced photosynthesis, weakened plants, reduced yield",
            "management": "Fungicides with copper compounds, resistant varieties, proper spacing for air circulation"
        },
        "Coffee Berry Disease": {
            "causative_agent": "Colletotrichum kahawae (fungus)",
            "symptoms": "Dark, sunken lesions on berries; brown/black spots on leaves",
            "effects": "Berry rot, premature fruit drop, significant yield loss",
            "management": "Copper-based fungicides, proper pruning, resistant varieties"
        },
        "Cercospora Leaf Spot": {
            "causative_agent": "Cercospora coffeicola (fungus)",
            "symptoms": "Brown spots with yellow halos on leaves; light-colored centers in advanced stages",
            "effects": "Defoliation, reduced plant vigor, lower yield quality",
            "management": "Fungicides, balanced nutrition, adequate spacing"
        },
        "Healthy": {
            "characteristics": "Vibrant green color, no spots or discoloration",
            "maintenance": "Regular monitoring, balanced fertilization, proper irrigation"
        }
    }
    
    # Use a safe get to handle unknown diseases
    info = disease_info.get(disease_name, {
        "note": "Information for this specific condition is not available."
    })
    
    if disease_name != "Healthy" and disease_name in disease_info:
        st.subheader(f"About {disease_name}")
        st.write(f"**Causative Agent:** {info['causative_agent']}")
        st.write(f"**Symptoms:** {info['symptoms']}")
        st.write(f"**Effects:** {info['effects']}")
        st.write(f"**Management:** {info['management']}")
    elif disease_name == "Healthy" and disease_name in disease_info:
        st.subheader("Healthy Leaf")
        st.write(f"**Characteristics:** {info['characteristics']}")
        st.write(f"**Maintenance:** {info['maintenance']}")
    else:
        st.subheader("Unknown Condition")
        st.write(info["note"])

# Function to find corresponding mask for an image
def find_mask_for_image(image_path, mask_dir):
    if not os.path.exists(mask_dir):
        return None
        
    image_name = os.path.basename(image_path)
    image_base, image_ext = os.path.splitext(image_name)
    
    # Try with same extension
    mask_path = os.path.join(mask_dir, image_name)
    if os.path.exists(mask_path):
        return mask_path
    
    # Try with different extensions
    for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
        alt_mask_path = os.path.join(mask_dir, image_base + ext)
        if os.path.exists(alt_mask_path):
            return alt_mask_path
            
    return None

# Function to display image processing steps
def display_image_processing(img, mask=None):
    try:
        # Convert PIL image to numpy array
        img_array = np.array(img)
        
        # Process mask if provided
        mask_array = None
        if mask is not None:
            try:
                mask_array = np.array(mask)
                # Convert to grayscale if it's RGB
                if len(mask_array.shape) == 3 and mask_array.shape[2] >= 3:
                    mask_array = cv2.cvtColor(mask_array, cv2.COLOR_RGB2GRAY)
                # Ensure binary mask
                _, mask_array = cv2.threshold(mask_array, 127, 255, cv2.THRESH_BINARY)
            except Exception as e:
                st.warning(f"Error processing mask: {str(e)}")
                logger.warning(f"Error processing mask: {e}")
                mask_array = None
        
        # Apply segmentation
        try:
            segmented_img = segment_leaf(img_array)
        except Exception as e:
            st.warning(f"Error in segmentation: {str(e)}")
            logger.warning(f"Error in segmentation: {e}")
            segmented_img = img_array.copy()
        
        # Apply CLAHE enhancement
        try:
            enhanced_img = apply_clahe(img_array)
        except Exception as e:
            st.warning(f"Error in enhancement: {str(e)}")
            logger.warning(f"Error in enhancement: {e}")
            enhanced_img = img_array.copy()
        
        # Extract features safely
        try:
            features = extract_features(img_array)
        except Exception as e:
            st.warning(f"Error extracting features: {str(e)}")
            logger.warning(f"Error extracting features: {e}")
            features = {"mean": 0, "std": 0, "hist": np.zeros(256)}
        
        # Display the processing steps
        st.subheader("Image Processing Steps")
        
        if mask_array is not None:
            # If mask is provided, show a 2x2 grid
            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)
            
            with col1:
                st.image(img_array, caption="Original Image", use_column_width=True)
            
            with col2:
                st.image(mask_array, caption="Ground Truth Mask", use_column_width=True)
                
            with col3:
                st.image(segmented_img, caption="Automated Segmentation", use_column_width=True)
                
            with col4:
                st.image(enhanced_img, caption="Enhanced Image (CLAHE)", use_column_width=True)
        else:
            # Without mask, show a 1x3 grid
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.image(img_array, caption="Original Image", use_column_width=True)
                
            with col2:
                st.image(segmented_img, caption="Leaf Segmentation", use_column_width=True)
                
            with col3:
                st.image(enhanced_img, caption="Enhanced Image (CLAHE)", use_column_width=True)
        
        # Display histogram of the image
        st.subheader("Image Histogram")
        try:
            fig, ax = plt.subplots(figsize=(10, 4))
            
            # RGB histograms
            colors = ('r', 'g', 'b')
            for i, color in enumerate(colors):
                hist = cv2.calcHist([img_array], [i], None, [256], [0, 256])
                ax.plot(hist, color=color, label=f'{color.upper()} Channel')
            
            ax.set_xlim([0, 256])
            ax.set_xlabel('Pixel Value')
            ax.set_ylabel('Frequency')
            ax.legend()
            
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error generating histogram: {str(e)}")
            logger.error(f"Error generating histogram: {e}")
        
        # Display feature information
        st.subheader("Extracted Features")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Mean Pixel Value:** {features.get('mean', 0):.2f}")
            st.write(f"**Standard Deviation:** {features.get('std', 0):.2f}")
        
        with col2:
            # Calculate some additional metrics
            if 'hist' in features and isinstance(features['hist'], (list, np.ndarray)):
                hist_array = np.array(features['hist'])
                if np.sum(hist_array) > 0:  # Avoid division by zero
                    peak = np.argmax(hist_array)
                    # Calculate entropy safely
                    hist_norm = hist_array / np.sum(hist_array)
                    entropy = -np.sum(np.where(hist_norm > 0, hist_norm * np.log2(hist_norm + 1e-10), 0))
                    
                    st.write(f"**Histogram Peak:** {peak}")
                    st.write(f"**Histogram Entropy:** {entropy:.2f}")
                else:
                    st.write("**Histogram analysis unavailable**")
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        logger.error(f"Error in display_image_processing: {e}")

# Function to show region of interest analysis
def show_roi_analysis(img, mask=None):
    try:
        img_array = np.array(img)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Try to detect edges or regions of interest
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours on original image
        contour_img = img_array.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
        
        # Highlight potential disease regions
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV) if len(img_array.shape) == 3 else np.zeros_like(img_array)
        
        # Define multiple color ranges for different diseases
        # Rust (orangish-yellow)
        lower_rust = np.array([15, 100, 100])
        upper_rust = np.array([30, 255, 255])
        rust_mask = cv2.inRange(hsv, lower_rust, upper_rust)
        
        # Berry disease (dark brown spots)
        lower_berry = np.array([0, 50, 20])
        upper_berry = np.array([10, 255, 100])
        berry_mask = cv2.inRange(hsv, lower_berry, upper_berry)
        
        # Cercospora (brown with yellow halo)
        lower_cercospora = np.array([20, 100, 100])
        upper_cercospora = np.array([40, 255, 255])
        cercospora_mask = cv2.inRange(hsv, lower_cercospora, upper_cercospora)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(rust_mask, berry_mask)
        combined_mask = cv2.bitwise_or(combined_mask, cercospora_mask)
        
        # Apply mask to image
        highlighted = img_array.copy()
        highlighted[combined_mask > 0] = [255, 0, 0]  # Highlight potential disease areas in red
        
        # Process mask if provided
        mask_array = None
        if mask is not None:
            try:
                mask_array = np.array(mask)
                # Convert to grayscale if it's RGB
                if len(mask_array.shape) == 3:
                    mask_array = cv2.cvtColor(mask_array, cv2.COLOR_RGB2GRAY)
                # Ensure binary mask
                _, mask_array = cv2.threshold(mask_array, 127, 255, cv2.THRESH_BINARY)
            except Exception as e:
                st.warning(f"Error processing ground truth mask: {str(e)}")
                logger.warning(f"Error processing ground truth mask: {e}")
        
        # Display the images
        st.subheader("Region of Interest Analysis")
        
        if mask_array is not None:
            # If mask is provided, show a comparison
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.image(contour_img, caption="Detected Edges & Contours", use_column_width=True)
            
            with col2:
                st.image(highlighted, caption="Potential Disease Regions", use_column_width=True)
                
            with col3:
                st.image(mask_array, caption="Ground Truth Mask", use_column_width=True)
                
            # Compare predicted vs ground truth
            try:
                # Calculate IoU or other metrics
                overlap = cv2.bitwise_and(combined_mask, mask_array)
                overlap_img = img_array.copy()
                overlap_img[overlap > 0] = [0, 255, 0]  # Green for overlapping areas
                
                st.subheader("Prediction vs Ground Truth")
                st.image(overlap_img, caption="Overlap Between Detected Regions and Ground Truth", use_column_width=True)
                
                # Calculate IoU
                intersection = np.sum(overlap > 0)
                union = np.sum(cv2.bitwise_or(combined_mask, mask_array) > 0)
                iou = intersection / union if union > 0 else 0
                
                st.write(f"**Intersection over Union (IoU):** {iou:.4f}")
            except Exception as e:
                st.warning(f"Error calculating overlap metrics: {str(e)}")
                logger.warning(f"Error calculating overlap metrics: {e}")
        else:
            # Without mask, just show detected regions
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(contour_img, caption="Detected Edges & Contours", use_column_width=True)
            
            with col2:
                st.image(highlighted, caption="Potential Disease Regions", use_column_width=True)
                
        # Show some statistics about the detected regions
        st.subheader("Region Statistics")
        
        # Number of contours found
        st.write(f"**Number of distinct regions detected:** {len(contours)}")
        
        # Calculate area covered by potential disease
        disease_area_percentage = np.sum(combined_mask > 0) / (combined_mask.shape[0] * combined_mask.shape[1]) * 100
        st.write(f"**Area potentially affected by disease:** {disease_area_percentage:.2f}%")
        
        # Show largest contours
        if len(contours) > 0:
            contour_areas = [cv2.contourArea(c) for c in contours]
            largest_contour_idx = np.argmax(contour_areas)
            largest_contour = contours[largest_contour_idx]
            
            # Draw just the largest contour
            largest_contour_img = img_array.copy()
            cv2.drawContours(largest_contour_img, [largest_contour], 0, (255, 0, 0), 2)
            
            st.write(f"**Largest region area:** {contour_areas[largest_contour_idx]:.2f} pixels")
            st.image(largest_contour_img, caption="Largest Detected Region", width=300)
            
    except Exception as e:
        st.error(f"Error in ROI analysis: {str(e)}")
        logger.error(f"Error in ROI analysis: {e}")

# Main application
def main():
    st.title("Coffee Leaf Disease Detection System")
    st.markdown("""
    This application uses deep learning to detect diseases in coffee plant leaves.
    Upload an image of a coffee leaf to get a diagnosis and information about potential diseases.
    """)
    
    # Load the model
    model = load_disease_model()
    if model is None:
        st.warning("Running in demo mode as model couldn't be loaded.")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        ["Upload Image", "View Examples", "About"]
    )
    
    # Upload image mode
    if app_mode == "Upload Image":
        st.sidebar.subheader("Upload")
        uploaded_file = st.sidebar.file_uploader("Choose a coffee leaf image", type=['jpg', 'jpeg', 'png'])
        
        # Optional mask upload for researchers
        show_advanced = st.sidebar.checkbox("Show Advanced Options", False)
        uploaded_mask = None
        
        if show_advanced:
            uploaded_mask = st.sidebar.file_uploader("Upload ground truth mask (optional)", type=['jpg', 'jpeg', 'png'])
            st.sidebar.info("Ground truth masks help evaluate the accuracy of segmentation and disease detection.")
        
        if uploaded_file is not None:
            # Display uploaded image
            try:
                image = Image.open(uploaded_file).convert('RGB')
                st.subheader("Uploaded Image")
                st.image(image, caption="Uploaded Image", width=300)
                
                # Process mask if provided
                mask = None
                if uploaded_mask is not None:
                    mask = Image.open(uploaded_mask).convert('L')  # Convert to grayscale
                
                # Make prediction if model is loaded
                if model is not None:
                    with st.spinner("Analyzing leaf..."):
                        # Get prediction
                        predicted_class, confidence, all_probabilities = predict_disease(image, model)
                        disease_name = CLASS_NAMES[predicted_class]
                        
                        # Display prediction
                        st.header("Diagnosis Results")
                        st.markdown(f"**Detected Condition:** {disease_name}")
                        st.markdown(f"**Confidence:** {confidence:.2%}")
                        
                        # Display confidence for all classes
                        st.subheader("Probability Distribution")
                        prob_df = {
                            "Disease": CLASS_NAMES,
                            "Probability": [float(p) for p in all_probabilities]
                        }
                        
                        # Create bar chart for probabilities
                        fig, ax = plt.subplots(figsize=(10, 4))
                        bars = ax.bar(
                            CLASS_NAMES,
                            [float(p) for p in all_probabilities],
                            color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
                        )
                        ax.set_ylabel('Probability')
                        ax.set_title('Disease Probability Distribution')
                        
                        # Add value labels on top of each bar
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(
                                bar.get_x() + bar.get_width()/2.,
                                height + 0.01,
                                f'{height:.2f}',
                                ha='center',
                                va='bottom'
                            )
                        
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Display disease information
                        display_disease_info(disease_name)
                
                # Image processing visualization
                with st.expander("View Image Processing Steps", expanded=False):
                    display_image_processing(image, mask)
                
                # ROI analysis
                with st.expander("View Region of Interest Analysis", expanded=False):
                    show_roi_analysis(image, mask)
                
            except Exception as e:
                st.error(f"Error processing the image: {str(e)}")
                logger.error(f"Error in main application flow: {e}")
        else:
            st.info("Please upload an image of a coffee leaf to get started.")
    
    # Example mode
    elif app_mode == "View Examples":
        st.subheader("Example Images")
        
        if not EXAMPLE_IMAGES_PATH or not os.path.exists(EXAMPLE_IMAGES_PATH):
            st.warning("Example images folder not found. Please check the path configuration.")
        else:
            try:
                # Get all example images
                examples = [f for f in os.listdir(EXAMPLE_IMAGES_PATH) if f.endswith(('.jpg', '.jpeg', '.png'))]
                
                if not examples:
                    st.warning("No example images found in the examples directory.")
                else:
                    # Create tabs for each disease category
                    categories = ["All"] + CLASS_NAMES
                    tabs = st.tabs(categories)
                    
                    # For each category tab
                    for i, category in enumerate(categories):
                        with tabs[i]:
                            if category == "All":
                                filtered_examples = examples
                            else:
                                filtered_examples = [ex for ex in examples if category.lower() in ex.lower()]
                            
                            if not filtered_examples:
                                st.write(f"No examples found for {category}")
                                continue
                            
                            # Display in a grid with 3 columns
                            cols = st.columns(3)
                            for j, example in enumerate(filtered_examples):
                                col_idx = j % 3
                                with cols[col_idx]:
                                    example_path = os.path.join(EXAMPLE_IMAGES_PATH, example)
                                    example_img = Image.open(example_path).convert('RGB')
                                    st.image(example_img, caption=example, use_column_width=True)
                                    
                                    # Add a button to analyze this example
                                    if st.button(f"Analyze This Image", key=f"analyze_{j}_{category}"):
                                        # Find corresponding mask if available
                                        example_mask_path = find_mask_for_image(
                                            example_path, 
                                            os.path.join(EXAMPLE_IMAGES_PATH, "masks") if os.path.exists(os.path.join(EXAMPLE_IMAGES_PATH, "masks")) else None
                                        )
                                        
                                        example_mask = None
                                        if example_mask_path and os.path.exists(example_mask_path):
                                            example_mask = Image.open(example_mask_path).convert('L')
                                        
                                        st.session_state.selected_image = example_img
                                        st.session_state.selected_mask = example_mask
                                        st.session_state.selected_image_name = example
                                        
                                        # This will trigger a page rerun and show the analysis below
                                        st.experimental_rerun()
                
                # Show analysis of selected image
                if hasattr(st.session_state, 'selected_image'):
                    st.subheader(f"Analysis of {getattr(st.session_state, 'selected_image_name', 'Example')}")
                    st.image(st.session_state.selected_image, caption="Selected Image", width=300)
                    
                    # Make prediction if model is loaded
                    if model is not None:
                        with st.spinner("Analyzing example leaf..."):
                            # Get prediction
                            predicted_class, confidence, all_probabilities = predict_disease(
                                st.session_state.selected_image, model
                            )
                            disease_name = CLASS_NAMES[predicted_class]
                            
                            # Display prediction
                            st.markdown(f"**Detected Condition:** {disease_name}")
                            st.markdown(f"**Confidence:** {confidence:.2%}")
                            
                            # Display disease information
                            display_disease_info(disease_name)
                    
                    # Image processing visualization
                    with st.expander("View Image Processing Steps", expanded=False):
                        display_image_processing(
                            st.session_state.selected_image, 
                            getattr(st.session_state, 'selected_mask', None)
                        )
                    
                    # ROI analysis
                    with st.expander("View Region of Interest Analysis", expanded=False):
                        show_roi_analysis(
                            st.session_state.selected_image, 
                            getattr(st.session_state, 'selected_mask', None)
                        )
                    
                    # Clear button
                    if st.button("Clear Analysis"):
                        del st.session_state.selected_image
                        del st.session_state.selected_image_name
                        if hasattr(st.session_state, 'selected_mask'):
                            del st.session_state.selected_mask
                        st.experimental_rerun()
                    
            except Exception as e:
                st.error(f"Error loading example images: {str(e)}")
                logger.error(f"Error in examples mode: {e}")
    
    # About mode
    elif app_mode == "About":
        st.subheader("About This Application")
        st.markdown("""
        ### Coffee Leaf Disease Detection System
        
        This application uses deep learning to detect diseases in coffee plant leaves. It can identify:
        
        - **Coffee Leaf Rust (CLR)**: Caused by the fungus *Hemileia vastatrix*, this is one of the most devastating coffee diseases worldwide.
        - **Coffee Berry Disease (CBD)**: Caused by *Colletotrichum kahawae*, affecting coffee berries and causing significant yield loss.
        - **Cercospora Leaf Spot**: Caused by *Cercospora coffeicola*, characterized by brown spots with yellow halos.
        - **Healthy Coffee Leaves**: Normal leaves without disease symptoms.
        
        ### Technologies Used
        - **TensorFlow/Keras**: For the deep learning model
        - **OpenCV**: For image processing and analysis
        - **Streamlit**: For the web application interface
        - **Matplotlib/Numpy**: For data visualization and numerical operations
        
        ### How It Works
        1. Upload an image of a coffee leaf
        2. The image is preprocessed and segmented
        3. A pre-trained deep learning model analyzes the leaf
        4. The application displays the diagnosis and additional information
        
        ### Data Source
        The model was trained on a dataset of coffee leaf images collected from various coffee farms, with expert annotations.
        
        ### About the Authors
        This application was developed by researchers dedicated to improving coffee cultivation through technology and AI solutions.
        """)

if __name__ == "__main__":
    main()
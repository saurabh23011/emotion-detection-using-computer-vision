import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import tempfile
import time

# Set page configuration
st.set_page_config(
    page_title="Emotion Detection App",
    page_icon="😀",
    layout="wide",
)

# Main title and description
st.title("Emotion Detection App")
st.markdown("""
This application uses a Convolutional Neural Network to detect and classify emotions in faces.
Upload an image or use your webcam to see it in action!
""")

# Sidebar for model explanation and parameters
with st.sidebar:
    st.header("About the Model")
    st.markdown("""
    This emotion detection model is trained on the FER2013 dataset and can recognize 7 emotions:
    - 😠 Angry
    - 🤢 Disgust
    - 😨 Fear
    - 😄 Happy
    - 😐 Neutral
    - 😢 Sad
    - 😲 Surprise
    
    The model uses a CNN architecture with multiple convolutional layers followed by batch normalization,
    max pooling, and dropout layers to prevent overfitting.
    """)
    
    # Add a confidence threshold slider
    confidence_threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5,
        help="Minimum confidence level to display a prediction"
    )
    
    # Add detection parameters
    st.header("Face Detection Parameters")
    scale_factor = st.slider("Scale Factor", min_value=1.1, max_value=2.0, value=1.2, step=0.1, 
                            help="Parameter specifying how much the image size is reduced at each image scale")
    min_neighbors = st.slider("Min Neighbors", min_value=1, max_value=10, value=5, 
                             help="Parameter specifying how many neighbors each candidate rectangle should have to retain it")
    min_face_size = st.slider("Min Face Size", min_value=10, max_value=100, value=30, 
                             help="Minimum possible object size. Objects smaller than this are ignored")

# Function to load the emotion detection model
@st.cache_resource
def load_model():
    try:
        # Load model architecture
        model_path = "Weights/network_emotions.json"
        weights_path = "Weights/weights_emotions.hdf5"
        
        with open(model_path, 'r') as json_file:
            loaded_model_json = json_file.read()
            
        model = model_from_json(loaded_model_json)
        model.load_weights(weights_path)
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to load the Haar cascade for face detection
@st.cache_resource
def load_face_detector():
    try:
        face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        return face_detector
    except Exception as e:
        st.error(f"Error loading face detector: {e}")
        return None

# Function to process the image and detect emotions
def process_image(image, face_detector, model, confidence_threshold, scale_factor, min_neighbors, min_face_size):
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    emotion_emoji = ['😠', '🤢', '😨', '😄', '😐', '😢', '😲']
    
    # Convert image to RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    
    # Create a copy of the image for drawing
    result_image = image.copy()
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(min_face_size, min_face_size)
    )
    
    face_results = []
    
    # Process each detected face
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Extract the face ROI
        face_roi = image[y:y + h, x:x + w]
        
        # Preprocess the face
        try:
            face_roi = cv2.resize(face_roi, (48, 48))
            face_roi = face_roi / 255.0
            face_roi = np.expand_dims(face_roi, axis=0)
            
            # Predict emotion
            prediction = model.predict(face_roi, verbose=0)
            
            # Get the emotion with highest probability
            max_index = np.argmax(prediction[0])
            confidence = prediction[0][max_index]
            
            # Only display if confidence is above threshold
            if confidence >= confidence_threshold:
                emotion = emotions[max_index]
                emoji = emotion_emoji[max_index]
                
                # Draw emotion label
                label = f"{emoji} {emotion}: {confidence:.2f}"
                cv2.putText(result_image, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Store face result for displaying statistics
                face_results.append({
                    'emotion': emotion,
                    'confidence': confidence,
                    'position': (x, y, w, h)
                })
        except Exception as e:
            st.error(f"Error processing face: {e}")
    
    return result_image, face_results, len(faces)

# Main function to run the app
def main():
    # Load the model and face detector
    model = load_model()
    face_detector = load_face_detector()
    
    if model is None or face_detector is None:
        st.error("Failed to load model or face detector. Please check if the model files are available.")
        return
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["Upload Image", "Use Webcam"])
    
    # Tab 1: Upload Image
    with tab1:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Read the image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Convert BGR to RGB for display
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image
            result_image, face_results, num_faces = process_image(
                image, face_detector, model, confidence_threshold, scale_factor, min_neighbors, min_face_size
            )
            
            # Convert result to RGB for display
            result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            
            # Display images side by side
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(image_rgb, use_column_width=True)
            
            with col2:
                st.subheader("Detected Emotions")
                st.image(result_rgb, use_column_width=True)
            
            # Display statistics
            st.subheader("Detection Statistics")
            st.write(f"Number of faces detected: {num_faces}")
            
            if face_results:
                # Create a table of detected emotions
                emotion_data = {
                    "Face": [f"Face {i+1}" for i in range(len(face_results))],
                    "Emotion": [f"{r['emotion']}" for r in face_results],
                    "Confidence": [f"{r['confidence']:.2f}" for r in face_results]
                }
                
                st.table(emotion_data)
            else:
                st.info("No emotions detected with confidence above threshold.")
    
    # Tab 2: Webcam
    with tab2:
        st.markdown("### Webcam Emotion Detection")
        
        # Start/stop button for webcam
        run = st.button("Start/Stop Webcam")
        
        # Placeholder for webcam feed
        stframe = st.empty()
        
        # Stats placeholder
        stats_placeholder = st.empty()
        
        # Store webcam state in session state
        if 'webcam_active' not in st.session_state:
            st.session_state.webcam_active = False
        
        if run:
            st.session_state.webcam_active = not st.session_state.webcam_active
        
        if st.session_state.webcam_active:
            try:
                # Open webcam
                cap = cv2.VideoCapture(0)
                
                # Check if webcam is opened successfully
                if not cap.isOpened():
                    st.error("Could not open webcam. Please check your camera settings.")
                    st.session_state.webcam_active = False
                else:
                    while st.session_state.webcam_active:
                        # Read frame
                        ret, frame = cap.read()
                        
                        if ret:
                            # Process frame
                            result_frame, face_results, num_faces = process_image(
                                frame, face_detector, model, confidence_threshold, scale_factor, min_neighbors, min_face_size
                            )
                            
                            # Convert to RGB for display
                            result_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                            
                            # Display the result
                            stframe.image(result_rgb, channels="RGB", use_column_width=True)
                            
                            # Display stats
                            stats_text = f"Number of faces detected: {num_faces}\n"
                            if face_results:
                                for i, result in enumerate(face_results):
                                    stats_text += f"Face {i+1}: {result['emotion']} ({result['confidence']:.2f})\n"
                            else:
                                stats_text += "No emotions detected with confidence above threshold."
                                
                            stats_placeholder.text(stats_text)
                            
                            # Add a small delay
                            time.sleep(0.01)
                        else:
                            st.error("Failed to read frame from webcam")
                            break
                    
                    # Release webcam
                    cap.release()
            except Exception as e:
                st.error(f"Error using webcam: {e}")
                st.session_state.webcam_active = False

if __name__ == "__main__":
    main()
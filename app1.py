import streamlit as st
import cv2 as cv
import numpy as np
import keras
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Disease labels and their treatments
label_name = ['Apple scab', 'Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy', 'Cherry Powdery mildew',
              'Cherry healthy', 'Corn Cercospora leaf spot Gray leaf spot', 'Corn Common rust', 'Corn Northern Leaf Blight', 'Corn healthy', 
              'Grape Black rot', 'Grape Esca', 'Grape Leaf blight', 'Grape healthy', 'Peach Bacterial spot', 'Peach healthy', 
              'Pepper bell Bacterial spot', 'Pepper bell healthy', 'Potato Early blight', 'Potato Late blight', 'Potato healthy', 
              'Strawberry Leaf scorch', 'Strawberry healthy', 'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight', 
              'Tomato Leaf Mold', 'Tomato Septoria leaf spot', 'Tomato Spider mites', 'Tomato Target Spot', 'Tomato Yellow Leaf Curl Virus', 
              'Tomato mosaic virus', 'Tomato healthy']

treatments = {
    'Apple scab': 'Use fungicides and plant resistant varieties. Remove infected leaves and prune trees for better air circulation.',
    'Apple Black rot': 'Prune infected branches and apply fungicide sprays during bloom and after petal fall.',
    'Apple Cedar apple rust': 'Remove nearby cedar trees if possible, as they host the fungus. Use fungicides like myclobutanil during the growing season.',
    'Apple healthy': 'No treatment needed. The plant is healthy!',
    
    'Cherry Powdery mildew': 'Prune to increase airflow, avoid overhead watering, and apply sulfur-based fungicides as needed.',
    'Cherry healthy': 'No treatment needed. The plant is healthy!',
    
    'Corn Cercospora leaf spot Gray leaf spot': 'Rotate crops, use resistant hybrids, and apply fungicides if severe.',
    'Corn Common rust': 'Plant resistant varieties, and consider fungicides if infection is heavy.',
    'Corn Northern Leaf Blight': 'Plant resistant varieties and apply fungicides at the first sign of infection if necessary.',
    'Corn healthy': 'No treatment needed. The plant is healthy!',
    
    'Grape Black rot': 'Remove infected leaves and fruit, prune for airflow, and use fungicides like myclobutanil.',
    'Grape Esca': 'Prune out infected areas early. Avoid wounds to minimize infection.',
    'Grape Leaf blight': 'Prune infected parts and apply fungicides like copper-based solutions.',
    'Grape healthy': 'No treatment needed. The plant is healthy!',
    
    'Peach Bacterial spot': 'Use copper-based bactericides early in the season and prune affected areas.',
    'Peach healthy': 'No treatment needed. The plant is healthy!',
    
    'Pepper bell Bacterial spot': 'Apply copper-based bactericides and practice crop rotation. Avoid overhead watering.',
    'Pepper bell healthy': 'No treatment needed. The plant is healthy!',
    
    'Potato Early blight': 'Use fungicides like chlorothalonil, and remove diseased plants promptly.',
    'Potato Late blight': 'Apply fungicides such as chlorothalonil or copper compounds. Remove infected plants and avoid wet foliage.',
    'Potato healthy': 'No treatment needed. The plant is healthy!',
    
    'Strawberry Leaf scorch': 'Remove affected leaves, avoid overhead watering, and apply fungicides as necessary.',
    'Strawberry healthy': 'No treatment needed. The plant is healthy!',
    
    'Tomato Bacterial spot': 'Use copper-based sprays and avoid overhead watering.',
    'Tomato Early blight': 'Apply fungicides like chlorothalonil or copper sprays and remove infected plant material.',
    'Tomato Late blight': 'Use fungicides like mancozeb, chlorothalonil, or copper sprays. Remove infected plants.',
    'Tomato Leaf Mold': 'Increase airflow, avoid overhead watering, and apply fungicides as needed.',
    'Tomato Septoria leaf spot': 'Remove infected leaves, ensure good airflow, and apply fungicides if needed.',
    'Tomato Spider mites': 'Use insecticidal soap, horticultural oil, or neem oil. Encourage natural predators like ladybugs.',
    'Tomato Target Spot': 'Apply fungicides such as chlorothalonil. Remove infected leaves and increase airflow.',
    'Tomato Yellow Leaf Curl Virus': 'Use virus-resistant tomato varieties and control whitefly populations that spread the virus.',
    'Tomato mosaic virus': 'Remove infected plants; there’s no cure. Sanitize tools and practice crop rotation.',
    'Tomato healthy': 'No treatment needed. The plant is healthy!'
}

model = keras.models.load_model('Training/model/Leaf Deases(96,88).h5')

# Custom CSS
st.markdown(
    """
    <style>
        body {
            background-image: url('C:\\Users\\moham\\OneDrive\\Desktop\\Courses_&_Projects\\AI_Projects\\Plant1\\plant-diseases-detect\\Media\\1.jpg'); /* Replace with your image URL */
            background-size: cover;
            background-color: black;
        }
        .title {
            text-align: center;
            font-size: 3em;
            font-weight: bold;
            color: #4CAF50;
        }
        .sidebar .sidebar-content {
            background-color: rgba(0, 0, 0, 0.7);
            color: #fff;
        }
        .footer {
            text-align: center;
            margin-top: 20px;
            font-size: 0.9em;
            color: #ccc;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# App Title
st.markdown('<h1 class="title">🌿 Plant Disease Detection</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
st.sidebar.markdown(
    """
    <ul>
        <li>📂 **Upload an Image**</li>
        <li>🔍 **Results**</li>
        <li>📧 **Email Recommendation**</li>
    </ul>
    """,
    unsafe_allow_html=True,
)

# Main Section
st.write(
    """
    **Instructions**: 
    - Upload images of **plant leaves** to detect potential diseases. 
    - Supported plants: Apple, Cherry, Corn, Grape, Peach, Pepper, Potato, Strawberry, Tomato.
    """
)

# File Uploader
uploaded_file = st.sidebar.file_uploader("📂 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    
    st.write("### Image Processing and Results")

    # Reading the uploaded image
    image_bytes = uploaded_file.read()
    img = cv.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv.IMREAD_COLOR)

    # Convert to grayscale for the first column
    grayscale_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, grayscale_image_encoded = cv.imencode('.jpg', grayscale_image)
    
    # Image Preprocessing
    normalized_image = np.expand_dims(cv.resize(cv.cvtColor(img, cv.COLOR_BGR2RGB), (150, 150)), axis=0)

    # Prediction
    st.write("### Predicting Disease...")
    predictions = model.predict(normalized_image)
    confidence = predictions[0] * 100
    predicted_label = label_name[np.argmax(predictions)]
    prediction_confidence = confidence[np.argmax(predictions)]
    
    # Creating the 2-column layout for the top row
    col1, col2 = st.columns(2)
    
    with col1:
        # Display grayscale image
        st.image(grayscale_image_encoded.tobytes(), caption="Grayscale Image", use_container_width=True)

    with col2:
        # Display 3D confidence plot
        st.write("### 3D Confidence Plot")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(np.arange(len(label_name)), confidence, c=confidence, cmap='viridis')
        ax.set_xlabel('Plant Diseases')
        ax.set_ylabel('Confidence (%)')
        ax.set_zlabel('Scores')
        ax.set_title('3D Confidence Plot for Prediction')
        st.pyplot(fig)

    # Creating the 2-column layout for the bottom row
    col3, col4 = st.columns(2)
    
    with col3:
        # Display predicted disease
        if prediction_confidence >= 80:
            st.markdown(f"<h2 style='color:green;'>🌟 Detected Disease: {predicted_label}</h2>", unsafe_allow_html=True)
            st.markdown(f"<h4>Confidence: {prediction_confidence:.2f}%</h4>", unsafe_allow_html=True)
            st.progress(int(prediction_confidence))
        else:
            st.error("Low confidence. Try another image.")
    
    with col4:
        # Display treatment recommendation
        st.write("### Recommended Treatment")
        treatment_info = treatments.get(predicted_label, "No specific treatment available.")
        st.info(treatment_info)
    
    # Email Functionality
    st.write("### Receive Treatment via Email")
    recipient_email = st.text_input("Enter your email:")
    if st.button("Send Recommendation"):
        if recipient_email:
            message = MIMEMultipart()
            message["Subject"] = f"Treatment Recommendation for {predicted_label}"
            message["From"] = "mohammedkhabab029@gmail.com"
            message["To"] = recipient_email
            body = f"Detected Disease: {predicted_label}\n\nTreatment Recommendation:\n{treatment_info}"
            message.attach(MIMEText(body, "plain"))
            try:
                with smtplib.SMTP("smtp.gmail.com", 587) as server:
                    server.starttls()
                    server.login("mohammedkhabab029@gmail.com", "mfke jcde zlxf nkdw")
                    server.sendmail(message["From"], message["To"], message.as_string())
                st.success("Email sent successfully!")
            except Exception as e:
                st.error(f"Failed to send email: {e}")
        else:
            st.warning("Please enter a valid email address.")
else:
    st.info("Upload a plant leaf image to start.")

# Footer
st.markdown('<p class="footer">© Five Seven IT Solutions Pvt Ltd | All rights reserved</p>', unsafe_allow_html=True)

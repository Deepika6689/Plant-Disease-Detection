import streamlit as st
import cv2 as cv
import numpy as np
import keras
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

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


# Streamlit UI
st.title("Plant Disease Detection")

st.write("Please input only leaf Images of Apple, Cherry, Corn, Grape, Peach, Pepper, Potato, Strawberry, and Tomato. Otherwise, the model may not work perfectly.")

# Load the model
model = keras.models.load_model('Training/model/Leaf Deases(96,88).h5')

# File uploader
uploaded_file = st.file_uploader("Upload an image")
if uploaded_file is not None:
    # Read image
    image_bytes = uploaded_file.read()
    img = cv.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv.IMREAD_COLOR)
    normalized_image = np.expand_dims(cv.resize(cv.cvtColor(img, cv.COLOR_BGR2RGB), (150, 150)), axis=0)
    
    # Prediction
    predictions = model.predict(normalized_image)
    st.image(image_bytes)
    predicted_label = label_name[np.argmax(predictions)]
    
    if predictions[0][np.argmax(predictions)] * 100 >= 80:
        st.write(f"Result: {predicted_label}")
        
        # Display treatment information
        treatment_info = treatments.get(predicted_label, "No specific treatment available.")
        st.write(f"Recommended treatment: {treatment_info}")
        
        # Email functionality
        recipient_email = st.text_input("Enter your email to receive the treatment recommendation:")
        if st.button("Send Treatment Recommendation"):
            if recipient_email:
                # Prepare the email content
                message = MIMEMultipart()
                message["Subject"] = f"Treatment Recommendation for {predicted_label}"
                message["From"] = "mohammedkhabab029@gmail.com"  # Replace with your email
                message["To"] = recipient_email
                body = f"Detected Disease: {predicted_label}\n\nTreatment Recommendation:\n{treatment_info}"
                message.attach(MIMEText(body, "plain"))
                
                # Send the email
                try:
                    with smtplib.SMTP("smtp.gmail.com", 587) as server:  # Replace with your SMTP server
                        server.starttls()
                        server.login("mohammedkhabab029@gmail.com", "mfke jcde zlxf nkdw")  # Replace with your login credentials
                        server.sendmail(message["From"], message["To"], message.as_string())
                    st.success("Email sent successfully!")
                except Exception as e:
                    st.error(f"Failed to send email: {e}")
            else:
                st.error("Please enter a valid email address.")
    else:
        st.write("Try another image. The model confidence was low.")

import firebase_admin
from firebase_admin import credentials, firestore
from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import SVC
import joblib
import mahotas
from flask import jsonify
import requests

# Initialize Flask app
app = Flask(__name__)

# Firebase setup
cred = credentials.Certificate('crop-52d66-firebase-adminsdk-9t44j-201df7bd3b.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load the trained SVM model, LabelEncoder, and MinMaxScaler
model = joblib.load('svm_model.pkl')
le = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl')

# Feature extraction functions
def rgb_bgr(image):
    """Converts an image from BGR to RGB."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def bgr_hsv(rgb_img):
    """Converts an image from RGB to HSV."""
    return cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)

def img_segmentation(rgb_img, hsv_img):
    """Segments the image based on healthy and diseased regions."""
    lower_green = np.array([25, 0, 20])
    upper_green = np.array([100, 255, 255])
    healthy_mask = cv2.inRange(hsv_img, lower_green, upper_green)

    lower_brown = np.array([10, 0, 10])
    upper_brown = np.array([30, 255, 255])
    disease_mask = cv2.inRange(hsv_img, lower_brown, upper_brown)

    final_mask = healthy_mask + disease_mask
    final_result = cv2.bitwise_and(rgb_img, rgb_img, mask=final_mask)
    return final_result

def fd_hu_moments(image):
    """Extracts Hu Moments features."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.HuMoments(cv2.moments(gray)).flatten()

def fd_haralick(image):
    """Extracts Haralick texture features."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return mahotas.features.haralick(gray).mean(axis=0)

def fd_histogram(image, mask=None):
    """Extracts color histogram features."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_image], [0, 1, 2], mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# Function to store prediction and retrieve solution from Firebase
def store_and_get_solution_from_firebase(prediction):
    # Store the prediction in Firebase
    prediction_ref = db.collection('disease_prediction').document()
    prediction_ref.set({
        'prediction': prediction
    })

    # Retrieve the corresponding solution from Firebase
    solution_ref = db.collection('disease_solution').document(prediction).get()

    if solution_ref.exists:
        solution_data = solution_ref.to_dict()
        return solution_data.get('solution')
    else:
        return 'No solution found'

def predict_disease(image_path):
    """Predicts the disease from the uploaded image."""
    image = cv2.imread(image_path)
    if image is None:
        return None

    image = cv2.resize(image, (500, 500))
    rgb_img = rgb_bgr(image)
    hsv_img = bgr_hsv(rgb_img)
    segmented_img = img_segmentation(rgb_img, hsv_img)

    fv_hu_moments = fd_hu_moments(segmented_img)
    fv_haralick = fd_haralick(segmented_img)
    fv_histogram = fd_histogram(segmented_img)

    global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
    global_feature = scaler.transform([global_feature])

    prediction = model.predict(global_feature)
    predicted_disease = le.inverse_transform(prediction)[0]

    # Store prediction in Firebase and retrieve the solution
    solution = store_and_get_solution_from_firebase(predicted_disease)

    return predicted_disease, solution

# Flask routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file uploaded', 400

        file = request.files['file']
        if file.filename == '':
            return 'No file selected', 400

        filepath = os.path.join('static', 'uploads', file.filename)
        file.save(filepath)

        # Perform disease prediction and fetch the solution
        predicted_disease, solution = predict_disease(filepath)

        return render_template('predict.html', prediction=predicted_disease, solution=solution)

    return render_template('predict.html')
@app.route('/handle_user_choice', methods=['POST'])
def handle_user_choice():
    user_choice = request.form.get('choice')
    prediction = request.form.get('prediction')
    solution = request.form.get('solution')

    # Generate a query for GPT based on user's selection
    query_mapping = {
        "What pest do we need to use?": f"For {prediction}, what pest control solutions should be considered?",
        "How to handle this?": f"How should I handle {prediction}? What steps should I take?",
        "Preventive measures?": f"What are the preventive measures for {prediction}?"
    }

    query = query_mapping.get(user_choice, "No additional information available.")

    # Send the query to GPT-3.5 and get a response
    gpt_response = get_gpt_solution(query)

    return render_template('predict.html', prediction=prediction, solution=solution, additional_solution=gpt_response)

def get_gpt_solution(query):
    """Function to send the query to GPT-3.5 and return the response."""
    openai_api_url = 'https://api.openai.com/v1/chat/completions'
    openai_api_key = "sk-your-api-key"  # Replace with your actual OpenAI API key

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {openai_api_key}'
    }

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query}
        ]
    }

    try:
        response = requests.post(openai_api_url, json=data, headers=headers)
        if response.status_code == 200:
            # Extract GPT's response
            gpt_answer = response.json().get('choices')[0]['message']['content']
            return gpt_answer
        else:
            return "Error: Could not retrieve solution from GPT."
    except Exception as e:
        return f"An error occurred: {str(e)}"


@app.route('/admin_register', methods=['GET', 'POST'])
def admin_register():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')

        # Store the admin credentials in Firestore
        admin_ref = db.collection('admins').document()
        admin_ref.set({
            'name': name,
            'email': email,
            'password': password  # Passwords should be hashed in a real application
        })

        return 'Admin registered successfully!'

    return render_template('register.html')

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        admin_ref = db.collection('admin').document('documnet')
        admin_ref.set({
            'email': email,
            'password': password  # Should ideally be hashed for security
        })

        return 'Admin login successfully!'

    return render_template('admin_login.html')

@app.route('/admin_feedback', methods=['GET', 'POST'])
def admin_feedback():
    if request.method == 'POST':
        name = request.form.get('name')
        location = request.form.get('location')
        experience = request.form.get('experience')
        crops = request.form.get('crops')
        satisfaction = request.form.get('satisfaction')
        improvements = request.form.get('improvements')

        feedback_ref = db.collection('feedback').document('feed')
        feedback_ref.set({
            'name': name,
            'location': location,
            'experience': experience,
            'crops': crops,
            'satisfaction': satisfaction,
            'improvements': improvements
        })

        return 'Thank you for your feedback!'

    return render_template('feedback.html')

@app.route('/view_feedbacks', methods=['GET'])
def view_feedbacks():
    feedbacks = []
    feedback_ref = db.collection('feedback').get()

    for doc in feedback_ref:
        feedbacks.append(doc.to_dict())

    return jsonify(feedbacks)

@app.route('/redirect')
def redirect_page():
    return render_template('redirect.html')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        user_input = request.form.get('user_input')
        openai_api_url = 'https://api.openai.com/v1/chat/completions'
        openai_api_key = "sk-your-api-key"  # Replace with actual OpenAI API key

        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_input}
            ]
        }

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {openai_api_key}'
        }

        response = requests.post(openai_api_url, json=data, headers=headers)

        if response.status_code == 200:
            bot_response = response.json().get('choices')[0]['message']['content']
        else:
            bot_response = "Error: Could not connect to OpenAI API"

        return render_template('chat.html', user_input=user_input, bot_response=bot_response)

    return render_template('chat.html')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
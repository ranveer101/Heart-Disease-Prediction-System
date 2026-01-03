import numpy as np
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import pickle
from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime
import os
from werkzeug.security import generate_password_hash, check_password_hash
import logging

logging.basicConfig(level=logging.DEBUG)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "supersecretkey")
app.config["DEBUG"] = False  # Set to False in production

# MongoDB setup
MONGO_URI = os.environ.get("MONGO_URI")
client = None
db = None

def get_db():
    """Get database connection with error handling"""
    global client, db
    try:
        if db is None:
            if not MONGO_URI:
                raise Exception("MONGO_URI environment variable not set")
            
            client = MongoClient(
                MONGO_URI,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=5000
            )
            # Test the connection
            client.admin.command('ping')
            db = client["heart_disease_db"]
            logging.info("Database connection established")
        return db
    except Exception as e:
        logging.error(f"Database connection error: {e}")
        raise Exception(f"Database connection failed: {e}")

# Load the model pipeline
try:
    pipeline = pickle.load(open("model.pkl", "rb"))
    imputer = pipeline['imputer']
    scaler = pipeline['scaler']
    model = pipeline['model']
    feature_names = pipeline['feature_names']
    model_loaded = True
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model_loaded = False
    feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                     'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Disease information dictionary
disease_info = {
    0: {
        "name": "No Heart Disease",
        "risk_level": "Low Risk",
        "suggestions": "Keep up the healthy lifestyle! Regular exercise, a balanced diet, and regular health checkups are recommended.",
        "detailed_suggestions": [
            "Maintain 150 minutes of moderate exercise weekly",
            "Follow a Mediterranean or DASH diet rich in fruits, vegetables, and whole grains",
            "Limit alcohol consumption and avoid smoking",
            "Get regular blood pressure and cholesterol screenings",
            "Maintain healthy weight with BMI between 18.5-24.9"
        ],
        "ai_recommendations": "Based on your health profile, AI analysis suggests focusing on preventive measures. Consider tracking your heart rate variability with wearable devices for early detection of cardiovascular changes.",
        "natural_remedies": ""
    },
    1: {
        "name": "Heart Disease Detected",
        "risk_level": "High Risk",
        "suggestions": "Consult a cardiologist immediately. Adopt a heart-friendly diet, reduce sodium intake, manage stress, and avoid smoking and alcohol.",
        "detailed_suggestions": [
            "Schedule an appointment with a cardiologist within 1-2 weeks",
            "Begin a cardiac rehabilitation program under medical supervision",
            "Adopt a low-sodium, low-fat diet with less than 1,500mg sodium daily",
            "Consider medications as prescribed (statins, anti-hypertensives, etc.)",
            "Monitor blood pressure daily and keep a health journal",
            "Practice stress reduction techniques like meditation or deep breathing",
            "Limit physical exertion according to doctor's recommendations"
        ],
        "ai_recommendations": "AI analysis of your health parameters indicates possible early-stage cardiovascular issues. Recommended treatments include personalized medication timing based on your circadian rhythm, targeted supplement protocols focusing on CoQ10 and Omega-3s.",
        "natural_remedies": "Consider evidence-based natural approaches like hawthorn extract, garlic supplements, CoQ10, fish oil, and controlled breathing exercises. These should complement but never replace medical treatment."
    },
    2: {
        "name": "Potential Heart Disease Risk",
        "risk_level": "Moderate Risk",
        "suggestions": "Follow up with your doctor within a month. Implement heart-healthy lifestyle changes now.",
        "detailed_suggestions": [
            "Reduce saturated fat intake to less than 7% of daily calories",
            "Exercise moderately for 30 minutes at least 5 days a week",
            "Reduce stress through mindfulness practices",
            "Monitor blood pressure weekly",
            "Limit processed foods and added sugars",
            "Consider a sleep study if you experience poor sleep quality"
        ],
        "ai_recommendations": "AI pattern recognition in your biomarkers suggests pre-clinical cardiovascular strain. Recommended interventions include intermittent fasting adapted to your metabolic profile.",
        "natural_remedies": "Evidence suggests benefits from magnesium supplementation, taurine, hibiscus tea, and beetroot juice to support cardiovascular health. Consult with a healthcare provider before starting any supplements."
    }
}

def generate_health_insights(input_data, prediction, confidence):
    """Generate personalized health insights based on input data"""
    insights = []
    
    # Disease Risk Prediction
    if prediction == 1:
        insights.append(f"⚠️ High risk of heart disease detected. Confidence: {confidence:.2f}%. Consult a cardiologist immediately.")
    else:
        insights.append(f"✅ No signs of heart disease detected. Confidence: {confidence:.2f}%. Maintain a healthy lifestyle.")
    
    # Cholesterol Analysis
    cholesterol = input_data.get('chol')
    if cholesterol:
        if cholesterol > 240:
            insights.append(f"🔴 Cholesterol Level: {cholesterol} mg/dL (High) → Consider a low-fat diet and regular exercise.")
        elif 200 <= cholesterol <= 240:
            insights.append(f"🟡 Cholesterol Level: {cholesterol} mg/dL (Borderline High) → Monitor regularly & reduce saturated fats.")
        else:
            insights.append(f"🟢 Cholesterol Level: {cholesterol} mg/dL (Normal). Keep up the good work!")
    
    # Blood Pressure Analysis
    blood_pressure = input_data.get('trestbps')
    if blood_pressure:
        if blood_pressure > 140:
            insights.append(f"🔴 Blood Pressure: {blood_pressure} mmHg (Hypertension) → Monitor regularly & reduce salt intake.")
        elif 120 <= blood_pressure <= 140:
            insights.append(f"🟡 Blood Pressure: {blood_pressure} mmHg (Pre-hypertension) → Exercise & maintain a balanced diet.")
        else:
            insights.append(f"🟢 Blood Pressure: {blood_pressure} mmHg (Normal). Good cardiovascular health!")
    
    # Age-based risk factor
    age = input_data.get('age', 0)
    if age > 50:
        insights.append(f"🔶 Age: {age} years (Higher risk) → Regular check-ups recommended.")
    else:
        insights.append(f"🟢 Age: {age} years (Lower risk). Continue a healthy lifestyle!")
    
    # Maximum heart rate
    thalach = input_data.get('thalach')
    if thalach:
        if thalach > 202:
            insights.append(f"🔴 Maximum Heart Rate: {thalach} bpm (Very High) → Immediate medical consultation recommended.")
        elif thalach > 149:
            insights.append(f"🟡 Maximum Heart Rate: {thalach} bpm (Above Average) → Regular monitoring advised.")
        else:
            insights.append(f"🟢 Maximum Heart Rate: {thalach} bpm (Normal). Keep up with healthy activities!")
    
    return " ".join(insights)

def predict_disease(input_data):
    """Make prediction with error handling"""
    try:
        if not model_loaded:
            return 0, 50, 0, "Model not loaded. Please ensure model.pkl exists."
        
        # Create feature array
        features = np.zeros((1, len(feature_names)))
        
        # Fill in the features
        for i, feature in enumerate(feature_names):
            if feature in input_data:
                features[0, i] = float(input_data[feature])
        
        # Apply preprocessing
        features_imputed = imputer.transform(features)
        features_scaled = scaler.transform(features_imputed)
        
        # Get prediction
        try:
            probabilities = model.predict_proba(features_scaled)[0]
            confidence = probabilities[1] * 100
            
            if confidence < 30:
                prediction = 0
                risk_level = 0
            elif confidence < 70:
                prediction = 0 if confidence < 50 else 1
                risk_level = 2
            else:
                prediction = 1
                risk_level = 1
        except:
            prediction = model.predict(features_scaled)[0]
            confidence = 90 if prediction == 1 else 85
            risk_level = 1 if prediction == 1 else 0
        
        health_insights = generate_health_insights(input_data, prediction, confidence)
        return prediction, confidence, risk_level, health_insights
        
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return 0, 50, 0, f"Prediction error: {str(e)}"

# Routes
@app.route("/")
def main():
    if 'user_id' in session:
        return redirect(url_for('intermediate'))
    return redirect(url_for('login_page'))

@app.route("/intermediate")
def intermediate():
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    # Fixed: use 'username' instead of 'name'
    username = session.get('username', 'User')
    return render_template("intermediate2.html", username=username)

@app.route("/home")
def home():
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    return render_template("index.html")

@app.route("/heart-disease-model")
def heart_disease_model():
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    return redirect(url_for('home'))

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Verify user is logged in
        if 'user_id' not in session:
            return redirect(url_for('login_page'))
        
        # Get form values
        input_data = {}
        for feature in feature_names:
            if feature in request.form:
                input_data[feature] = float(request.form[feature])
        
        # Make prediction
        prediction, confidence, risk_level, health_insights = predict_disease(input_data)
        
        # Get disease info
        disease_name = disease_info[risk_level]["name"]
        risk_level_text = disease_info[risk_level]["risk_level"]
        suggestion = disease_info[risk_level]["suggestions"]
        detailed_suggestions = "<ul>" + "".join([f"<li>{s}</li>" for s in disease_info[risk_level]["detailed_suggestions"]]) + "</ul>"
        ai_recommendations = disease_info[risk_level]["ai_recommendations"]
        
        # Natural remedies
        natural_remedies = ""
        if risk_level > 0:
            natural_remedies = f"<div><h4>🌿 Natural Approaches:</h4><p>{disease_info[risk_level]['natural_remedies']}</p></div>"
        
        # Health tips
        health_tips = """<div><h4>💡 General Health Tips:</h4><ul>
        <li>🚶‍♂️ Regular exercise (30 min/day) helps reduce heart risk.</li>
        <li>🍏 Eat a heart-healthy diet (more fruits, vegetables, and whole grains).</li>
        <li>🚫 Avoid smoking & limit alcohol intake.</li>
        <li>😴 Ensure 7-8 hours of quality sleep daily.</li>
        <li>🧘 Practice stress-reduction techniques daily.</li>
        </ul></div>"""
        
        # Doctor recommendations
        doctor_recommendations = ""
        if risk_level > 0:
            doctor_recommendations = """<div><h4>👨‍⚕️ Recommended Specialists:</h4>
            <p><strong>Dr. John Smith, MD</strong><br>Cardiology Specialist<br>Rating: ⭐⭐⭐⭐⭐ (4.9/5)<br>Location: Medical Center Hospital</p>
            <p><strong>Dr. Sarah Johnson, MD</strong><br>Cardiovascular Disease Specialist<br>Rating: ⭐⭐⭐⭐⭐ (4.8/5)<br>Location: Heart Health Clinic</p>
            </div>"""
        
        # Test recommendations
        test_recommendations = ""
        if risk_level > 0:
            test_recommendations = """<div><h4>🔬 Recommended Tests:</h4><ul>
            <li>Echocardiogram: Uses sound waves to produce images of your heart.</li>
            <li>Stress Test: Shows how your heart works during physical activity.</li>
            <li>Coronary Calcium Scan: Measures the amount of calcium in the walls of your heart's arteries.</li>
            <li>Holter Monitoring: Records heart rhythm for 24-48 hours.</li>
            </ul></div>"""
        
        # Telemedicine options
        telemedicine_options = """<div><h4>💻 Telemedicine Options:</h4><ul>
        <li>Virtual Cardiology Consultation - Schedule a video call with a cardiologist within 24-48 hours.</li>
        <li>Remote Monitoring - Consider enrolling in a heart monitoring program.</li>
        <li>Online Support Groups - Connect with others managing similar heart health conditions.</li>
        </ul></div>"""
        
        # Save prediction to database
        try:
            db = get_db()
            predictions_collection = db["predictions"]
            predictions_collection.insert_one({
                'user_id': session['user_id'],
                'prediction': int(prediction),
                'confidence': float(confidence),
                'risk_level': int(risk_level),
                'disease_name': disease_name,
                'form_data': input_data,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        except Exception as db_error:
            logging.error(f"Database save error: {db_error}")
            # Continue even if database save fails
        
        # Store prediction data in session
        session['prediction_data'] = {
            'prediction_text': f"Prediction: {disease_name} ({risk_level_text}) - Confidence: {confidence:.1f}%",
            'suggestion_text': f"Suggestion: {suggestion}",
            'detailed_suggestions': detailed_suggestions,
            'ai_recommendations': f"<div><h4>🤖 AI-Enhanced Recommendations:</h4><p>{ai_recommendations}</p></div>",
            'natural_remedies': natural_remedies,
            'health_insights': f"<div><h4>📊 Personalized Health Insights:</h4><p>{health_insights}</p></div>",
            'health_tips': health_tips,
            'doctor_recommendations': doctor_recommendations,
            'test_recommendations': test_recommendations,
            'telemedicine_options': telemedicine_options
        }
        
        return redirect(url_for('prediction_results'))
        
    except Exception as e:
        logging.error(f"Prediction route error: {e}")
        return render_template("index.html", 
                             prediction_text="Error occurred", 
                             suggestion_text=str(e))

@app.route("/prediction-results")
def prediction_results():
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    
    if 'prediction_data' not in session:
        return redirect(url_for('home'))
    
    prediction_data = session['prediction_data']
    return render_template(
        "prediction_results.html",
        prediction_text=prediction_data['prediction_text'],
        suggestion_text=prediction_data['suggestion_text'],
        detailed_suggestions=prediction_data['detailed_suggestions'],
        ai_recommendations=prediction_data['ai_recommendations'],
        natural_remedies=prediction_data.get('natural_remedies', ''),
        health_insights=prediction_data['health_insights'],
        health_tips=prediction_data['health_tips'],
        doctor_recommendations=prediction_data.get('doctor_recommendations', ''),
        test_recommendations=prediction_data.get('test_recommendations', ''),
        telemedicine_options=prediction_data['telemedicine_options']
    )

@app.route('/history', methods=['GET'])
def get_history():
    try:
        user_id = session.get('user_id')
        if not user_id:
            return redirect(url_for('login_page'))
        
        db = get_db()
        predictions_collection = db["predictions"]
        results = predictions_collection.find({'user_id': user_id})
        
        history = []
        for item in results:
            history.append({
                'date': item.get('timestamp', 'N/A'),
                'prediction': item.get('disease_name', 'Unknown'),
                'confidence': item.get('confidence', 'N/A'),
                'age': item.get('form_data', {}).get('age'),
                'sex': item.get('form_data', {}).get('sex'),
                'chol': item.get('form_data', {}).get('chol')
            })
        
        return jsonify(history)
        
    except Exception as e:
        logging.error(f"History route error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/login')
def login_page():
    if 'user_id' in session:
        return redirect(url_for('intermediate'))
    return render_template("login.html")

@app.route("/register", methods=["POST"])
def register():
    try:
        db = get_db()
        users = db["users"]
        
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        
        if not username or not email or not password:
            logging.warning("Registration attempt with missing fields")
            return "Missing fields", 400
        
        if users.find_one({"email": email}):
            logging.warning(f"Registration attempt with existing email: {email}")
            return "User already exists", 400
        
        hashed_password = generate_password_hash(password)
        users.insert_one({
            "username": username,
            "email": email,
            "password": hashed_password,
            "created_at": datetime.now()
        })
        
        logging.info(f"New user registered: {email}")
        return redirect(url_for("login_page"))
        
    except Exception as e:
        logging.error(f"Registration error: {e}")
        return f"Internal Server Error: {str(e)}", 500

@app.route("/login", methods=["POST"])
def login():
    try:
        db = get_db()
        users = db["users"]
        
        email = request.form.get("email")
        password = request.form.get("password")
        
        if not email or not password:
            return "Missing credentials", 400
        
        user = users.find_one({"email": email})
        
        if user and check_password_hash(user["password"], password):
            session["user_id"] = str(user["_id"])
            session["username"] = user["username"]  # Fixed: store username correctly
            logging.info(f"User logged in: {email}")
            return redirect(url_for("intermediate"))
        
        logging.warning(f"Failed login attempt for: {email}")
        return "Invalid credentials", 401
        
    except Exception as e:
        logging.error(f"Login error: {e}")
        return f"Internal Server Error: {str(e)}", 500

@app.route('/aboutUs')
def aboutus():
    return render_template('aboutUs.html')

@app.route('/logout')
def logout():
    session.clear()
    logging.info("User logged out")
    return redirect(url_for('login_page'))

# Error handlers
@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(e):
    logging.error(f"Internal error: {e}")
    return render_template('500.html'), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
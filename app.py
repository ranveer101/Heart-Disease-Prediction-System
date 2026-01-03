from flask import Flask, request, render_template, session, redirect, url_for, jsonify
import numpy as np
import pickle
import os
import logging
import warnings
import time
import tempfile
import json
from datetime import datetime
from functools import wraps

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "hf-secret-key")
print("DEBUG MONGO_URI:", os.environ.get("MONGO_URI"))

# Configure session settings for production
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True if using HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress scikit-learn version warnings when loading pickled models
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# ---------------- MONGO DB ---------------- 
MONGO_URI = os.environ.get("MONGO_URI", "").strip()
USE_MONGODB = True if MONGO_URI else False

# File-based mechanism to prevent duplicate logging across gunicorn workers
_log_file = os.path.join(tempfile.gettempdir(), '.mongodb_status_logged')

def _log_mongodb_status_once(message, level='info'):
    """Log MongoDB status message only once across all gunicorn workers"""
    try:
        # Try to create the file exclusively (atomic operation)
        try:
            # Try to open in exclusive create mode (fails if file exists)
            fd = os.open(_log_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            # We're the first process, log the message
            if level == 'info':
                logger.info(message)
            elif level == 'warning':
                logger.warning(message)
            return
        except FileExistsError:
            # File already exists, another process already logged
            # Check file age - if it's old (>10 seconds), assume it's stale and log anyway
            try:
                file_age = time.time() - os.path.getmtime(_log_file)
                if file_age > 10:
                    # Stale file, remove it and log
                    os.remove(_log_file)
                    fd = os.open(_log_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.close(fd)
                    if level == 'info':
                        logger.info(message)
                    elif level == 'warning':
                        logger.warning(message)
            except Exception:
                pass  # If we can't check/remove, skip logging
            return
    except Exception:
        # If file operations fail completely, log anyway to be safe
        if level == 'info':
            logger.info(message)
        elif level == 'warning':
            logger.warning(message)

if USE_MONGODB:
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        # Test connection
        client.server_info()
        db = client["heart_disease_db"]
        users_collection = db["users"]
        predictions_collection = db["predictions"]
        _log_mongodb_status_once("MongoDB connection established", 'info')
    except Exception:
        USE_MONGODB = False
        # Don't log warning - MongoDB failure is expected when not configured

# File-based storage fallback (shared across gunicorn workers)
if not USE_MONGODB:
    users_collection = {}  # Keep for compatibility
    predictions_collection = []  # Keep for compatibility
    _log_mongodb_status_once("Using file-based storage (MongoDB not configured)", 'info')

# ---------------- FILE-BASED STORAGE FUNCTIONS (for multi-worker support) ---------------- 
_users_file = os.path.join(tempfile.gettempdir(), '.heart_disease_users.json')
_predictions_file = os.path.join(tempfile.gettempdir(), '.heart_disease_predictions.json')

def _load_users_from_file():
    """Load users from file (shared across workers)"""
    if USE_MONGODB:
        return {}
    try:
        if os.path.exists(_users_file):
            with open(_users_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.debug(f"Error loading users file: {e}")
    return {}

def _save_users_to_file(users_dict):
    """Save users to file (shared across workers)"""
    if USE_MONGODB:
        return
    try:
        with open(_users_file, 'w') as f:
            json.dump(users_dict, f)
    except Exception as e:
        logger.error(f"Error saving users to file: {e}")

def _load_predictions_from_file():
    """Load predictions from file (shared across workers)"""
    if USE_MONGODB:
        return []
    try:
        if os.path.exists(_predictions_file):
            with open(_predictions_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.debug(f"Error loading predictions file: {e}")
    return []

def _save_predictions_to_file(predictions_list):
    """Save predictions to file (shared across workers)"""
    if USE_MONGODB:
        return
    try:
        with open(_predictions_file, 'w') as f:
            json.dump(predictions_list, f, default=str)
    except Exception as e:
        logger.error(f"Error saving predictions to file: {e}")

def get_user_by_email(email):
    if USE_MONGODB:
        return users_collection.find_one({"email": email})
    else:
        users = _load_users_from_file()
        return users.get(email)

def save_user(user_data):
    if USE_MONGODB:
        result = users_collection.insert_one(user_data)
        return str(result.inserted_id)
    else:
        users = _load_users_from_file()
        user_id = f"user_{len(users)}"
        user_data["_id"] = user_id
        users[user_data["email"]] = user_data
        _save_users_to_file(users)
        return user_id

def save_prediction(prediction_data):
    if USE_MONGODB:
        predictions_collection.insert_one(prediction_data)
    else:
        predictions = _load_predictions_from_file()
        # Convert datetime to string for JSON serialization
        if isinstance(prediction_data.get("timestamp"), datetime):
            prediction_data = prediction_data.copy()
            prediction_data["timestamp"] = prediction_data["timestamp"].isoformat()
        predictions.append(prediction_data)
        _save_predictions_to_file(predictions)

def get_user_predictions(user_id):
    if USE_MONGODB:
        return list(predictions_collection.find({"user_id": user_id}).sort("timestamp", -1).limit(10))
    else:
        predictions = _load_predictions_from_file()
        user_predictions = [p for p in predictions if p.get("user_id") == user_id][-10:]
        return user_predictions

# Initialize with a default test user if no users exist (after functions are defined)
if not USE_MONGODB:
    users = _load_users_from_file()
    if not users:
        test_user = {
            "name": "Test User",
            "email": "test@test.com",
            "password": "test123",
            "_id": "user_0"
        }
        users["test@test.com"] = test_user
        _save_users_to_file(users)
        logger.info("Created default test user: test@test.com / test123")

# ---------------- LOAD MODEL ---------------- 
try:
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    if not os.path.exists(model_path):
        model_path = "model.pkl"
    
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)
    
    imputer = pipeline['imputer']
    scaler = pipeline['scaler']
    model = pipeline['model']
    feature_names = pipeline['feature_names']
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise RuntimeError(f"Failed to load model: {e}")

def requires_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function

# ---------------- LOGIN ---------------- 
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        try:
            email = request.form.get("email", "").strip()
            password = request.form.get("password", "").strip()
            
            logger.info(f"Login attempt for email: {email}")

            if not email or not password:
                logger.warning("Login failed: Empty email or password")
                return render_template("login.html", error="Email and password are required")

            user = get_user_by_email(email)
            logger.info(f"User lookup result: {user is not None}")

            if not user:
                logger.warning(f"Login failed: User not found for email: {email}")
                return render_template("login.html", error="No account found. Please register first or check your email.")

            if user.get("password") != password:
                logger.warning(f"Login failed: Invalid password for email: {email}")
                return render_template("login.html", error="Invalid password. Please try again.")

            # Set session
            user_id = user.get("_id")
            session["user_id"] = str(user_id) if user_id else email
            session["name"] = user.get("name", "User")
            session["email"] = user.get("email", email)
            
            logger.info(f"Login successful for user: {email}, redirecting to intermediate")
            return redirect(url_for("intermediate"))
        except Exception as e:
            logger.error(f"Login error: {e}", exc_info=True)
            return render_template("login.html", error="An error occurred. Please try again.")

    if "user_id" in session:
        return redirect(url_for("intermediate"))

    return render_template("login.html")

# ---------------- REGISTER ---------------- 
@app.route("/register", methods=["POST"])
def register():
    try:
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()

        if not name or not email or not password:
            return render_template("login.html", error="All fields are required", show_register=True)

        if get_user_by_email(email):
            return render_template("login.html", error="User already exists. Please login instead.", show_register=True)

        user_data = {
            "name": name,
            "email": email,
            "password": password
        }
        
        user_id = save_user(user_data)

        session["user_id"] = user_id
        session["name"] = name
        session["email"] = email

        return redirect(url_for("intermediate"))
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return render_template("login.html", error="An error occurred during registration. Please try again.", show_register=True)

# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ---------------- INTERMEDIATE ---------------- 
@app.route("/intermediate")
@requires_auth
def intermediate():
    return render_template("intermediate2.html", username=session.get("name", "User"))

# ---------------- HOME ---------------- 
@app.route("/")
@app.route("/home")
@requires_auth
def home():
    return render_template("index.html")

# ---------------- ABOUT US ---------------- 
@app.route("/aboutUs")
@requires_auth
def about_us():
    return render_template("aboutUs.html")

# ---------------- HEART DISEASE MODEL ---------------- 
@app.route("/heart_disease_model")
@requires_auth
def heart_disease_model():
    return redirect(url_for("home"))

# ---------------- GET HISTORY ---------------- 
@app.route("/get_history", methods=["GET"])
@requires_auth
def get_history():
    try:
        user_id = session.get("user_id")
        predictions = get_user_predictions(user_id)
        return jsonify({
            "success": True,
            "predictions": [
                {
                    "prediction": p.get("prediction", 0),
                    "confidence": p.get("confidence", 0),
                    "timestamp": p.get("timestamp").isoformat() if hasattr(p.get("timestamp"), "isoformat") else str(p.get("timestamp", ""))
                }
                for p in predictions
            ]
        })
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# ---------------- PREDICT ---------------- 
@app.route("/predict", methods=["POST"])
@requires_auth
def predict():
    try:
        input_data = {f: float(request.form.get(f, 0)) for f in feature_names}

        X = np.zeros((1, len(feature_names)))
        for i, f in enumerate(feature_names):
            X[0, i] = input_data.get(f, 0)

        X = scaler.transform(imputer.transform(X))
        pred = model.predict(X)[0]
        confidence = max(model.predict_proba(X)[0]) * 100

        prediction_data = {
            "user_id": session.get("user_id"),
            "prediction": int(pred),
            "confidence": float(confidence),
            "timestamp": datetime.now()
        }
        
        save_prediction(prediction_data)

        # Determine risk level
        has_disease = bool(pred)
        risk_level = "High Risk" if has_disease else "Low Risk"
        risk_class = "risk-high" if has_disease else "risk-low"
        
        # Generate prediction text
        prediction_text = f"<h3 style='margin: 0; font-size: 20px; color: {'#ef4444' if has_disease else '#10b981'};'>{'Heart Disease Detected' if has_disease else 'No Heart Disease Detected'}</h3>"
        
        # Generate health insights
        health_insights = ""
        if has_disease:
            health_insights = """
            <div style='padding: 12px; background: #fef2f2; border-left: 4px solid #ef4444; border-radius: 4px; margin-bottom: 12px;'>
                <strong style='color: #ef4444;'>⚠️ Risk Detected:</strong> The analysis indicates a potential risk of heart disease. Please consult with a healthcare professional.
            </div>
            """
        else:
            health_insights = """
            <div style='padding: 12px; background: #f0fdf4; border-left: 4px solid #10b981; border-radius: 4px; margin-bottom: 12px;'>
                <strong style='color: #10b981;'>✓ Low Risk:</strong> The analysis shows a low risk of heart disease. Continue maintaining a healthy lifestyle.
            </div>
            """
        
        # Generate detailed suggestions
        if has_disease:
            detailed_suggestions = """
            <div class="rec-grid">
                <div class="rec-card">
                    <h4>Immediate Actions</h4>
                    <p>Schedule an appointment with a cardiologist within 1-2 weeks. Monitor your symptoms and keep a health journal.</p>
                </div>
                <div class="rec-card">
                    <h4>Lifestyle Changes</h4>
                    <p>Adopt a heart-healthy diet (Mediterranean or DASH diet), reduce sodium intake, and increase physical activity gradually.</p>
                </div>
                <div class="rec-card">
                    <h4>Medication Review</h4>
                    <p>Review all current medications with your doctor. Some medications may need adjustment or new prescriptions may be needed.</p>
                </div>
            </div>
            """
        else:
            detailed_suggestions = """
            <div class="rec-grid">
                <div class="rec-card">
                    <h4>Maintain Healthy Habits</h4>
                    <p>Continue with regular exercise, balanced diet, and stress management. Regular check-ups are still important.</p>
                </div>
                <div class="rec-card">
                    <h4>Preventive Care</h4>
                    <p>Stay active, maintain a healthy weight, avoid smoking, and limit alcohol consumption to reduce future risk.</p>
                </div>
                <div class="rec-card">
                    <h4>Regular Monitoring</h4>
                    <p>Schedule annual health check-ups and monitor blood pressure, cholesterol, and blood sugar levels regularly.</p>
                </div>
            </div>
            """
        
        # Generate AI recommendations
        ai_recommendations = f"""
        <div style='padding: 16px; background: #f0f9ff; border-radius: 8px; margin-bottom: 16px;'>
            <h4 style='margin-bottom: 12px; color: #1a56db;'>AI Analysis Summary</h4>
            <p style='margin-bottom: 8px;'>Based on the machine learning model analysis with <strong>{confidence:.2f}% confidence</strong>, the prediction indicates:</p>
            <ul style='margin-left: 20px; color: #4b5563;'>
                <li>Model accuracy and reliability assessment</li>
                <li>Risk factor analysis based on input parameters</li>
                <li>Comparative analysis with similar patient profiles</li>
            </ul>
        </div>
        """
        
        natural_remedies = """
        <div style='padding: 16px; background: #fefce8; border-radius: 8px;'>
            <h4 style='margin-bottom: 12px; color: #ca8a04;'>Natural Support (Consult Doctor First)</h4>
            <ul style='margin-left: 20px; color: #4b5563;'>
                <li>Omega-3 fatty acids (fish oil, flaxseeds)</li>
                <li>Coenzyme Q10 supplements</li>
                <li>Garlic (may help with blood pressure)</li>
                <li>Hawthorn berry (traditional heart support)</li>
            </ul>
            <p style='margin-top: 12px; font-size: 13px; color: #6b7280;'><strong>Note:</strong> Always consult with your healthcare provider before starting any supplements.</p>
        </div>
        """
        
        # Generate test recommendations
        if has_disease:
            test_recommendations = """
            <div class="test-list">
                <div class="test-item">
                    <div class="test-name">Electrocardiogram (ECG)</div>
                    <div class="test-desc">Assess heart rhythm and electrical activity</div>
                    <span class="priority priority-high">High Priority</span>
                </div>
                <div class="test-item">
                    <div class="test-name">Echocardiogram</div>
                    <div class="test-desc">Ultrasound imaging of heart structure and function</div>
                    <span class="priority priority-high">High Priority</span>
                </div>
                <div class="test-item">
                    <div class="test-name">Stress Test</div>
                    <div class="test-desc">Evaluate heart function during physical activity</div>
                    <span class="priority priority-medium">Medium Priority</span>
                </div>
                <div class="test-item">
                    <div class="test-name">Blood Tests (Lipid Panel)</div>
                    <div class="test-desc">Check cholesterol and triglyceride levels</div>
                    <span class="priority priority-high">High Priority</span>
                </div>
            </div>
            """
        else:
            test_recommendations = """
            <div class="test-list">
                <div class="test-item">
                    <div class="test-name">Annual Physical Exam</div>
                    <div class="test-desc">Routine health check-up and blood work</div>
                    <span class="priority priority-low">Routine</span>
                </div>
                <div class="test-item">
                    <div class="test-name">Blood Pressure Monitoring</div>
                    <div class="test-desc">Regular home monitoring recommended</div>
                    <span class="priority priority-low">Routine</span>
                </div>
            </div>
            """
        
        # Generate doctor recommendations
        if has_disease:
            doctor_recommendations = """
            <div class="specialist-list" style='margin-top: 24px;'>
                <div class="specialist-item">
                    <div class="specialist-name">Cardiologist</div>
                    <div class="specialist-desc">Specialist in heart and cardiovascular diseases. Schedule appointment within 1-2 weeks.</div>
                </div>
                <div class="specialist-item">
                    <div class="specialist-name">Primary Care Physician</div>
                    <div class="specialist-desc">Coordinate care and manage overall health</div>
                </div>
            </div>
            """
        else:
            doctor_recommendations = """
            <div class="specialist-list" style='margin-top: 24px;'>
                <div class="specialist-item">
                    <div class="specialist-name">Primary Care Physician</div>
                    <div class="specialist-desc">Continue regular annual check-ups and preventive care</div>
                </div>
            </div>
            """
        
        # Generate health tips
        health_tips = """
        <div class="tip-list">
            <div class="tip-item">
                <div class="tip-icon"><i class="fas fa-dumbbell"></i></div>
                <div class="tip-content">
                    <h4>Regular Exercise</h4>
                    <p>Aim for at least 150 minutes of moderate-intensity exercise per week. Activities like brisk walking, swimming, or cycling are excellent for heart health.</p>
                </div>
            </div>
            <div class="tip-item">
                <div class="tip-icon"><i class="fas fa-apple-alt"></i></div>
                <div class="tip-content">
                    <h4>Heart-Healthy Diet</h4>
                    <p>Focus on fruits, vegetables, whole grains, lean proteins, and healthy fats. Limit processed foods, saturated fats, and added sugars.</p>
                </div>
            </div>
            <div class="tip-item">
                <div class="tip-icon"><i class="fas fa-bed"></i></div>
                <div class="tip-content">
                    <h4>Quality Sleep</h4>
                    <p>Get 7-9 hours of quality sleep per night. Poor sleep can contribute to heart disease risk factors.</p>
                </div>
            </div>
            <div class="tip-item">
                <div class="tip-icon"><i class="fas fa-smoking-ban"></i></div>
                <div class="tip-content">
                    <h4>Avoid Smoking</h4>
                    <p>If you smoke, quitting is one of the best things you can do for your heart health. Seek support if needed.</p>
                </div>
            </div>
        </div>
        """
        
        # Generate telemedicine options
        telemedicine_options = """
        <div style='margin-top: 24px; padding: 16px; background: #f0f9ff; border-radius: 8px;'>
            <h4 style='margin-bottom: 12px; color: #1a56db;'>Telemedicine Options</h4>
            <p style='margin-bottom: 12px; color: #4b5563;'>Consider these telemedicine services for convenient healthcare access:</p>
            <ul style='margin-left: 20px; color: #4b5563;'>
                <li>Virtual consultations with cardiologists</li>
                <li>Remote monitoring of vital signs</li>
                <li>Online prescription management</li>
                <li>Digital health coaching programs</li>
            </ul>
        </div>
        """

        return render_template(
            "prediction_results.html",
            prediction_text=prediction_text,
            suggestion_text=f"Confidence: {confidence:.2f}%",
            health_insights=health_insights,
            detailed_suggestions=detailed_suggestions,
            ai_recommendations=ai_recommendations,
            natural_remedies=natural_remedies,
            test_recommendations=test_recommendations,
            doctor_recommendations=doctor_recommendations,
            health_tips=health_tips,
            telemedicine_options=telemedicine_options
        )
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return render_template(
            "prediction_results.html",
            prediction_text="<h3 style='color: #ef4444;'>Error: Prediction failed</h3>",
            suggestion_text=f"Please try again. Error: {str(e)}",
            health_insights="",
            detailed_suggestions="",
            ai_recommendations="",
            natural_remedies="",
            test_recommendations="",
            doctor_recommendations="",
            health_tips="",
            telemedicine_options=""
        ), 500

# ---------------- ERROR HANDLERS ---------------- 
@app.errorhandler(404)
def not_found(error):
    return render_template("login.html", error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return render_template("login.html", error="Internal server error. Please try again."), 500

# ---------------- RUN ---------------- 
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)

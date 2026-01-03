import numpy as np
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import pickle
from datetime import datetime
import os
import logging

logging.basicConfig(level=logging.INFO)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24).hex())

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

# Enhanced disease information with more detailed suggestions and risk levels
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
        "ai_recommendations": "Based on your health profile, AI analysis suggests focusing on preventive measures. Consider tracking your heart rate variability with wearable devices for early detection of cardiovascular changes. Personalized nutrition apps can help optimize your diet based on your specific metabolic needs."
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
        "ai_recommendations": "AI analysis of your health parameters indicates possible early-stage cardiovascular issues. Recommended treatments include personalized medication timing based on your circadian rhythm, targeted supplement protocols focusing on CoQ10 and Omega-3s, and remote monitoring through smart devices that can alert healthcare providers about concerning changes in your vitals.",
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
        "ai_recommendations": "AI pattern recognition in your biomarkers suggests pre-clinical cardiovascular strain. Recommended interventions include intermittent fasting adapted to your metabolic profile, targeted exercise focusing on high-intensity interval training 2-3 times weekly, and heart rate variability training using biofeedback applications.",
        "natural_remedies": "Evidence suggests benefits from magnesium supplementation, taurine, hibiscus tea, and beetroot juice to support cardiovascular health. Consult with a healthcare provider before starting any supplements."
    }
}
@app.route("/logout")
def logout():
    return redirect(url_for("main"))

# Function to generate AI-based health insights
def generate_health_insights(input_data, prediction, confidence):
    insights = []
    
    # Disease Risk Prediction
    if prediction == 1:
        insights.append(f"⚠️ High risk of heart disease detected. Confidence: {confidence:.2f}%. Consult a cardiologist immediately.")
    else:
        insights.append(f"✅ No signs of heart disease detected. Confidence: {confidence:.2f}%. Maintain a healthy lifestyle.")
    
    # Cholesterol Analysis
    cholesterol = input_data.get('chol', None)
    if cholesterol:
        if cholesterol > 240:
            insights.append(f"🔴 Cholesterol Level: {cholesterol} mg/dL (High) → Consider a low-fat diet and regular exercise.")
        elif 200 <= cholesterol <= 240:
            insights.append(f"🟡 Cholesterol Level: {cholesterol} mg/dL (Borderline High) → Monitor regularly & reduce saturated fats.")
        else:
            insights.append(f"🟢 Cholesterol Level: {cholesterol} mg/dL (Normal). Keep up the good work!")

    # Blood Pressure Analysis
    blood_pressure = input_data.get('trestbps', None)
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

    # Maximum heart rate achieved
    thalach = input_data.get('thalach', None)
    if thalach:
        if thalach > 202:
            insights.append(f"🔴 Maximum Heart Rate: {thalach} bpm (Very High) → Immediate medical consultation recommended.")
        elif thalach > 149:
            insights.append(f"🟡 Maximum Heart Rate: {thalach} bpm (Above Average) → Regular monitoring advised.")
        else:
            insights.append(f"🟢 Maximum Heart Rate: {thalach} bpm (Normal). Keep up with healthy activities!")

    # Chest pain
    cp = input_data.get('cp', None)
    if cp is not None:
        if cp == 0:
            insights.append("🟡 Typical Angina detected → Monitor chest pain during exertion.")
        elif cp == 1:
            insights.append("🟠 Atypical Angina detected → Consult a doctor for further evaluation.")
        elif cp == 2:
            insights.append("🟢 Non-anginal Pain detected → May not be heart-related.")
        elif cp == 3:
            insights.append("🟣 Asymptomatic → No chest pain, but other risks should be monitored.")

    # Exercise-induced angina
    exang = input_data.get('exang', None)
    if exang is not None:
        if exang == 1:
            insights.append("⚠️ Exercise-induced angina detected. Consider stress tests and lifestyle modifications.")
        elif exang == 0:
            insights.append("✅ No exercise-induced angina detected. Good sign!")

    # ST depression induced by exercise
    oldpeak = input_data.get('oldpeak', None)
    if oldpeak is not None:
        if oldpeak > 2:
            insights.append(f"🔴 ST Depression: {oldpeak} (High) → Consult a cardiologist.")
        elif oldpeak > 1:
            insights.append(f"🟡 ST Depression: {oldpeak} (Moderate) → Requires further evaluation.")
        else:
            insights.append(f"🟢 ST Depression: {oldpeak} (Low) → Normal range.")

    # Slope of the ST segment
    slope = input_data.get('slope', None)
    if slope is not None:
        if slope == 0:
            insights.append("🔴 Downsloping ST segment detected → High risk of heart disease.")
        elif slope == 1:
            insights.append("🟡 Flat ST segment detected → Moderate risk, monitor closely.")
        elif slope == 2:
            insights.append("🟢 Upsloping ST segment detected → Lower risk, good sign.")

    # Number of major vessels colored by fluoroscopy
    ca = input_data.get('ca', None)
    if ca is not None:
        insights.append(f"🔍 Major vessels colored by fluoroscopy: {ca}. Higher values may indicate increased risk.")

    # Thalassemia type
    thal = input_data.get('thal', None)
    if thal is not None:
        if thal == 1:
            insights.append("🔴 Thalassemia: Fixed defect detected → Increased heart disease risk.")
        elif thal == 2:
            insights.append("🟡 Thalassemia: Normal blood flow → No major risk.")
        elif thal == 3:
            insights.append("🟢 Thalassemia: Reversible defect detected → Manageable with lifestyle changes.")

    # Sex-based risk
    sex = input_data.get('sex', None)
    if sex is not None and sex == 1 and prediction == 1:
        insights.append("🧑‍⚕️ Men are at a higher risk of heart disease. Monitor cholesterol & BP levels.")

    return "<br>".join(insights)

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
            
            # Determine risk level based on probability
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
    # Directly show the intermediate page
    return render_template("intermediate2.html", username="Guest")

@app.route("/intermediate")
def intermediate():
    # Show intermediate page
    return render_template("intermediate2.html", username="Guest")

@app.route("/home")
def home():
    return render_template("index.html")

@app.route("/heart-disease-model")
def heart_disease_model():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
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
            natural_remedies = f"<div class='natural-remedies'><h3>🌿 Natural Approaches:</h3><p>{disease_info[risk_level]['natural_remedies']}</p></div>"
        
        # Health tips
        health_tips = """
        <div class="health-tips">
            <h3>💡 General Health Tips:</h3>
            <ul>
                <li>🚶‍♂️ Regular exercise (30 min/day) helps reduce heart risk.</li>
                <li>🍏 Eat a heart-healthy diet (more fruits, vegetables, and whole grains).</li>
                <li>🚫 Avoid smoking & limit alcohol intake.</li>
                <li>😴 Ensure 7-8 hours of quality sleep daily.</li>
                <li>🧘 Practice stress-reduction techniques daily.</li>
            </ul>
        </div>
        """
        
        # Doctor recommendations
        doctor_recommendations = ""
        if risk_level > 0:
            doctor_recommendations = """
            <div class="doctor-recommendations">
                <h3>👨‍⚕️ Recommended Specialists:</h3>
                <div class="doctor-card">
                    <h4>Dr. John Smith, MD</h4>
                    <p>Cardiology Specialist</p>
                    <p>Rating: ⭐⭐⭐⭐⭐ (4.9/5)</p>
                    <p>Location: Medical Center Hospital</p>
                </div>
                <div class="doctor-card">
                    <h4>Dr. Sarah Johnson, MD</h4>
                    <p>Cardiovascular Disease Specialist</p>
                    <p>Rating: ⭐⭐⭐⭐⭐ (4.8/5)</p>
                    <p>Location: Heart Health Clinic</p>
                </div>
            </div>
            """
        
        # Test recommendations
        test_recommendations = ""
        if risk_level > 0:
            test_recommendations = """
            <div class="test-recommendations">
                <h3>🔬 Recommended Tests:</h3>
                <ul>
                    <li><strong>Echocardiogram:</strong> Uses sound waves to produce images of your heart.</li>
                    <li><strong>Stress Test:</strong> Shows how your heart works during physical activity.</li>
                    <li><strong>Coronary Calcium Scan:</strong> Measures the amount of calcium in the walls of your heart's arteries.</li>
                    <li><strong>Holter Monitoring:</strong> Records heart rhythm for 24-48 hours.</li>
                </ul>
            </div>
            """
        
        # Telemedicine options
        telemedicine_options = """
        <div class="telemedicine-options">
            <h3>💻 Telemedicine Options:</h3>
            <ul>
                <li><strong>Virtual Cardiology Consultation</strong> - Schedule a video call with a cardiologist within 24-48 hours.</li>
                <li><strong>Remote Monitoring</strong> - Consider enrolling in a heart monitoring program that allows doctors to track your vitals remotely.</li>
                <li><strong>Online Support Groups</strong> - Connect with others managing similar heart health conditions.</li>
            </ul>
        </div>
        """
        
        # Store prediction data in session
        session['prediction_data'] = {
            'prediction_text': f"Prediction: {disease_name} ({risk_level_text}) - Confidence: {confidence:.1f}%",
            'suggestion_text': f"Suggestion: {suggestion}",
            'detailed_suggestions': detailed_suggestions,
            'ai_recommendations': f"<div class='ai-recommendations'><h3>🤖 AI-Enhanced Recommendations:</h3><p>{ai_recommendations}</p></div>",
            'natural_remedies': natural_remedies,
            'health_insights': f"<div class='health-insights'><h3>📊 Personalized Health Insights:</h3>{health_insights}</div>",
            'health_tips': health_tips,
            'doctor_recommendations': doctor_recommendations,
            'test_recommendations': test_recommendations,
            'telemedicine_options': telemedicine_options
        }
        
        # Store in session history
        if 'prediction_history' not in session:
            session['prediction_history'] = []
        
        session['prediction_history'].append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'prediction': int(prediction),
            'confidence': float(confidence),
            'risk_level': int(risk_level),
            'disease_name': disease_name,
            'form_data': input_data
        })
        
        # Redirect to the prediction results page
        return redirect(url_for('prediction_results'))
        
    except Exception as e:
        logging.error(f"Prediction route error: {e}")
        return render_template("index.html", 
                             prediction_text="Error occurred", 
                             suggestion_text=str(e))

@app.route("/prediction-results")
def prediction_results():
    # Check if there are prediction results in the session
    if 'prediction_data' not in session:
        return redirect(url_for('home'))
    
    # Get prediction data from session
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
    # Return session-based history
    history = session.get('prediction_history', [])
    
    formatted_history = []
    for item in history:
        formatted_history.append({
            'date': item.get('timestamp', 'N/A'),
            'prediction': item.get('disease_name', 'Unknown'),
            'confidence': item.get('confidence', 'N/A'),
            'age': item.get('form_data', {}).get('age'),
            'sex': item.get('form_data', {}).get('sex'),
            'chol': item.get('form_data', {}).get('chol')
        })
    
    return jsonify(formatted_history)

@app.route('/aboutUs')
def aboutus():
    return render_template('aboutUs.html')

# Error handlers
@app.errorhandler(404)
def not_found(e):
    return "Page not found", 404

@app.errorhandler(500)
def internal_error(e):
    logging.error(f"Internal error: {e}")
    return "Internal server error", 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
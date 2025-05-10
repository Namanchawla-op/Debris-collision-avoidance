from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from prediction_model import predict_damage_location_and_plane
from decision_model import evaluate_collisions_and_decisions

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Root route: render the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route: handles POST requests with satellite parameters
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Parse input parameters from the request
        semi_major = float(data.get('semiMajorAxis', 0))
        eccentricity = float(data.get('eccentricity', 0))
        inclination = float(data.get('inclination', 0))

        # Build full satellite data dictionary
        satellite_data = {
            'SEMIMAJOR_AXIS': semi_major,
            'ECCENTRICITY': eccentricity,
            'INCLINATION': inclination,
            'RA_OF_ASC_NODE': 0,
            'ARG_OF_PERICENTER': 0,
            'MEAN_ANOMALY': 0,
            'MEAN_MOTION': 0,
            'VELOCITY_PERIGEE': 0,
            'VELOCITY_APOGEE': 0
        }

        # Run prediction model
        damage_plane, damage_assessment, probability = predict_damage_location_and_plane(satellite_data)

        # Run decision model for maneuver recommendations
        decision_results = evaluate_collisions_and_decisions([
            semi_major, eccentricity, inclination, 0, 0, 0, 0, 0, 0
        ])

        # Return results to frontend
        return jsonify({
            'probability': probability,  # already scaled in percentage
            'damage_plane': damage_plane,
            'damage_assessment': damage_assessment,
            'recommendations': decision_results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Entry point to run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
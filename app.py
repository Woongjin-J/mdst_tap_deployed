import numpy as np
import torch
import torch.nn as nn
import folium
from folium.plugins import MousePosition
from folium import plugins
import json
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import os
import socket
import webbrowser
from threading import Timer
import requests
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_port_available(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(('127.0.0.1', port))
        available = True
    except:
        available = False
    sock.close()
    return available

def find_available_port(start_port=8080):
    port = start_port
    while not check_port_available(port) and port < start_port + 20:
        port += 1
    return port if port < start_port + 20 else None

# Define the neural network model for accident severity prediction
class AccidentSeverityModel(nn.Module):
    def __init__(self):
        super(AccidentSeverityModel, self).__init__()
        # Define the layers
        self.fc1 = nn.Linear(9, 128)  # Input layer with 9 features, outputting 128 nodes
        self.fc2 = nn.Linear(128, 64)  # Hidden layer with 64 nodes
        self.fc3 = nn.Linear(64, 4)  # Output layer predicting one of four severity levels
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # Dropout layer for regularization
        nn.init.xavier_uniform_(self.fc1.weight)  # Initialize weights for better training
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Load the pre-trained model weights
try:
    model = AccidentSeverityModel()
    model.load_state_dict(torch.load('best_accident_severity_model.pth'))
    model.eval()  # Set model to evaluation mode
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please make sure 'accident_severity_model.pth' is in the current directory.")
    exit(1)


# Create Flask app
app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

def get_road_info(lat, lon):
    """
    Get road type and name using Overpass API
    """
    try:
        # Define the Overpass query to get nearby roads with names
        overpass_url = "http://overpass-api.de/api/interpreter"
        query = f"""
        [out:json];
        way(around:50,{lat},{lon})
        ["highway"];
        out body;
        >;
        out body qt;
        """

        logger.info(f"Querying Overpass API for location: {lat}, {lon}")
        response = requests.post(overpass_url, data=query)

        if response.status_code != 200:
            logger.warning(f"Overpass API returned status code: {response.status_code}")
            return {'type': 'Unknown', 'name': 'Unknown'}

        data = response.json()

        if not data.get('elements'):
            logger.info("No roads found nearby")
            return {'type': 'Unknown', 'name': 'Unknown'}

        # Check all nearby roads and categorize them
        roads = []
        for element in data['elements']:
            if 'tags' in element and 'highway' in element['tags']:
                road_info = {
                    'highway_type': element['tags']['highway'],
                    'name': element['tags'].get('name', 'Unnamed Road'),
                    'ref': element['tags'].get('ref', '')  # Reference number for highways
                }
                roads.append(road_info)

        logger.info(f"Found roads: {roads}")

        # Categorize the road types
        highway_categories = {
            'motorway': 'Highway',
            'trunk': 'Highway',
            'motorway_link': 'Highway',
            'trunk_link': 'Highway',
            'primary': 'Highway',
            'primary_link': 'Highway',
            'secondary': 'Highway',
            'secondary_link': 'Highway'
        }

        # First, try to find a highway
        for road in roads:
            if road['highway_type'] in highway_categories:
                road_name = road['ref'] if road['ref'] else road['name']
                return {'type': 'Highway', 'name': road_name}

        # If no highway found but we have roads, return the first named road
        if roads:
            road = roads[0]
            return {'type': 'Local Road', 'name': road['name']}

        return {'type': 'Unknown', 'name': 'Unknown'}

    except requests.exceptions.RequestException as e:
        logger.error(f"Error querying Overpass API: {str(e)}")
        return {'type': 'Unknown', 'name': 'Unknown'}
    except Exception as e:
        logger.error(f"Unexpected error in get_road_info: {str(e)}")
        return {'type': 'Unknown', 'name': 'Unknown'}


def open_browser(port):
    webbrowser.open(f'http://127.0.0.1:{port}/')


@app.route('/')
def home():
    try:
        return send_file('accident_prediction_map.html')
    except Exception as e:
        return f"Error: Could not load the HTML file. {str(e)}", 500


@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 204
    try:
        data = request.json
        lat, lon = data.get('lat'), data.get('lon')

        # Get road info if coordinates are provided
        road_info = {'type': 'Unknown', 'name': 'Unknown'}
        if lat is not None and lon is not None:
            logger.info(f"Getting road info for coordinates: {lat}, {lon}")
            road_info = get_road_info(lat, lon)
            logger.info(f"Detected road info: {road_info}")
            # Update Highway_Flag based on road type
            data['Highway_Flag'] = 1 if road_info['type'] == 'Highway' else 0

        transformed = transform_features(data)
        x = torch.tensor([list(transformed.values())], dtype=torch.float32)
        with torch.no_grad():
            output = model(x)
            pred = torch.argmax(output, dim=1).item() + 1

        response_data = {
            'severity': pred,
            'road_type': road_info['type'],
            'road_name': road_info['name'],
            'is_highway': road_info['type'] == 'Highway'
        }
        logger.info(f"Prediction response: {response_data}")
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500


# Define initial feature inputs (default values)
features = {
    'Traffic_Signal_Flag': 0,
    'Crossing_Flag': 0,
    'Highway_Flag': 1,
    'Distance(mi)': 1.0,
    'Start_Hour': 12,
    'Start_Month': 6,
    'Accident_Duration': 5
}


def transform_features(raw):
    # Transform raw inputs to a tensor suitable for prediction
    return {
        'Traffic_Signal_Flag': raw['Traffic_Signal_Flag'],
        'Crossing_Flag': raw['Crossing_Flag'],
        'Highway_Flag': raw['Highway_Flag'],
        'Distance(mi)': raw['Distance(mi)'],
        'Start_Hour_Sin': np.sin(2 * np.pi * raw['Start_Hour'] / 24),
        'Start_Hour_Cos': np.cos(2 * np.pi * raw['Start_Hour'] / 24),
        'Start_Month_Sin': np.sin(2 * np.pi * raw['Start_Month'] / 12),
        'Start_Month_Cos': np.cos(2 * np.pi * raw['Start_Month'] / 12),
        'Accident_Duration': (raw['Accident_Duration'] - 315.48991736325416) / 9888.00371222839  # Normalized duration
    }


# Initialize a Folium map centered over the USA
m = folium.Map(location=[37.0902, -95.7129], zoom_start=4)


# Function to handle click events on the map
def on_click(location):
    lat, lon = location  # Extract latitude and longitude from click event
    severity = predict_severity()  # Predict severity using the model
    folium.Marker(
        location=[lat, lon],
        popup=f"Predicted Severity: {severity}",  # Display prediction as popup text
        icon=folium.Icon(color='red')  # Marker is colored red for visibility
    ).add_to(m)


# Add a plugin to display latitude and longitude of the mouse position
formatter = "function(num) {return L.Util.formatNum(num, 5);};"
MousePosition(position="bottomleft", separator=" | ", empty_string="Unavailable", lng_first=True, num_digits=5, prefix="Coordinates:", lat_formatter=formatter, lng_formatter=formatter).add_to(m)


# Create HTML template with sliders and controls
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Accident Severity Prediction</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
            display: flex;
            height: 100vh;
            overflow: hidden;
        }
        #map {
            flex: 70%;
            height: 100%;
        }
        #controls {
            flex: 30%;
            padding: 15px;
            background-color: #f5f5f5;
            height: 100%;
            overflow-y: auto;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        .toggle-container {
            margin-bottom: 0;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 8px 12px;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .toggle-label {
            font-weight: bold;
            flex: 1;
        }
        .toggle-switch {
            position: relative;
            display: inline-block;
            width: 46px;
            height: 26px;
        }
        .toggle-switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }
        .toggle-slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: #ccc;
            transition: .4s;
            border-radius: 34px;
        }
        .toggle-slider:before {
            position: absolute;
            content: "";
            height: 20px;
            width: 20px;
            left: 3px;
            bottom: 3px;
            background-color: white;
            transition: .4s;
            border-radius: 50%;
        }
        input:checked + .toggle-slider {
            background-color: #2196F3;
        }
        input:checked + .toggle-slider:before {
            transform: translateX(20px);
        }
        .slider-container {
            margin-bottom: 0;
            padding: 8px 12px;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .slider-label {
            display: block;
            margin-bottom: 4px;
            font-weight: bold;
            font-size: 0.95em;
        }
        .slider {
            width: 100%;
            margin: 8px 0;
        }
        .slider-value {
            display: inline-block;
            margin-left: 8px;
            font-weight: bold;
            font-size: 0.9em;
        }
        #prediction {
            margin: 0;
            padding: 12px;
            background-color: #fff;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        #prediction h3 {
            margin-top: 0;
            margin-bottom: 8px;
            font-size: 1.2em;
        }
        #prediction p {
            margin: 6px 0;
            font-size: 0.95em;
        }
        h2 {
            margin: 0 0 4px 0;
            font-size: 1.4em;
        }
        p {
            margin: 0 0 12px 0;
            font-size: 0.95em;
        }
        .severity-high {
            color: red;
            font-weight: bold;
        }
        .severity-medium {
            color: orange;
            font-weight: bold;
        }
        .severity-low {
            color: green;
            font-weight: bold;
        }
        #error-message {
            display: none;
            background-color: #ffebee;
            color: #c62828;
            padding: 10px;
            margin-top: 10px;
            border-radius: 4px;
        }
        .road-type {
            margin-top: 10px;
            padding: 8px;
            background-color: #e3f2fd;
            border-radius: 4px;
            font-weight: bold;
        }
        .highway {
            color: #1976d2;
        }
        .local-road {
            color: #388e3c;
        }
        .unknown-road {
            color: #757575;
        }
        .loading {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.5);
            z-index: 9999;
            justify-content: center;
            align-items: center;
        }
        .loading.show {
            display: flex;
        }
        .loading-content {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .spinner {
            width: 24px;
            height: 24px;
            border: 3px solid #ff9800;
            border-radius: 50%;
            border-top-color: transparent;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div id="loading" class="loading">
        <div class="loading-content">
            <div class="spinner"></div>
            <span>Detecting road type...</span>
        </div>
    </div>
    <div id="controls">
        <h2>Accident Severity Prediction</h2>
        <p>Click on the map to predict accident severity at that location</p>
        <div id="error-message"></div>
        <div id="prediction">
            <h3>Current Prediction</h3>
            <p>Click on the map to see prediction</p>
        </div>
        <div class="toggle-container">
            <span class="toggle-label">Traffic Signal</span>
            <label class="toggle-switch">
                <input type="checkbox" id="Traffic_Signal_Flag">
                <span class="toggle-slider"></span>
            </label>
        </div>
        <div class="toggle-container">
            <span class="toggle-label">Crossing</span>
            <label class="toggle-switch">
                <input type="checkbox" id="Crossing_Flag">
                <span class="toggle-slider"></span>
            </label>
        </div>
        <div class="toggle-container">
            <span class="toggle-label">Highway (Auto-detected)</span>
            <label class="toggle-switch">
                <input type="checkbox" id="Highway_Flag" checked>
                <span class="toggle-slider"></span>
            </label>
        </div>
        <div class="slider-container">
            <label class="slider-label">Distance (miles)</label>
            <input type="range" min="0" max="0.5" value="0.1" class="slider" id="Distance(mi)" step="0.01">
            <span class="slider-value">0.1</span>
        </div>
        <div class="slider-container">
            <label class="slider-label">Start Hour</label>
            <input type="range" min="0" max="23" value="12" class="slider" id="Start_Hour" step="1">
            <span class="slider-value">12</span>
        </div>
        <div class="slider-container">
            <label class="slider-label">Start Month</label>
            <input type="range" min="1" max="12" value="6" class="slider" id="Start_Month" step="1">
            <span class="slider-value">6</span>
        </div>
        <div class="slider-container">
            <label class="slider-label">Accident Duration (minutes)</label>
            <input type="range" min="1" max="180" value="5" class="slider" id="Accident_Duration" step="1">
            <span class="slider-value">5</span>
        </div>
    </div>
    <div id="map"></div>
    <script>
        // Initialize the map
        var map = L.map('map').setView([37.0902, -95.7129], 4);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(map);

        var currentMarker = null;
        var features = {
            'Traffic_Signal_Flag': 0,
            'Crossing_Flag': 0,
            'Highway_Flag': 1,
            'Distance(mi)': 1.0,
            'Start_Hour': 12,
            'Start_Month': 6,
            'Accident_Duration': 5
        };

        let updateTimer = null;

        function showError(message) {
            const errorDiv = document.getElementById('error-message');
            errorDiv.textContent = message;
            errorDiv.style.display = 'block';
            setTimeout(() => {
                errorDiv.style.display = 'none';
            }, 5000);
        }

        function updateRoadTypeDisplay(roadType, roadName) {
            // Update Highway Flag toggle only
            const highwayToggle = document.getElementById('Highway_Flag');
            const newValue = roadType === 'Highway' ? 1 : 0;
            highwayToggle.checked = newValue === 1;
            features['Highway_Flag'] = newValue;
        }

        // Update toggle values and handle changes
        document.querySelectorAll('.toggle-switch input').forEach(toggle => {
            toggle.addEventListener('change', function() {
                features[this.id] = this.checked ? 1 : 0;

                // Clear any existing timer
                if (updateTimer) {
                    clearTimeout(updateTimer);
                }

                // Set a new timer to update prediction after 500ms
                if (currentMarker) {
                    updateTimer = setTimeout(() => {
                        updatePrediction(currentMarker.getLatLng());
                    }, 500);
                }
            });
        });

        // Update slider values display and handle changes
        document.querySelectorAll('.slider').forEach(slider => {
            const valueDisplay = slider.nextElementSibling;
            valueDisplay.textContent = slider.value;

            slider.addEventListener('input', function() {
                const value = this.value;
                valueDisplay.textContent = value;
                features[this.id] = parseFloat(value);

                // Clear any existing timer
                if (updateTimer) {
                    clearTimeout(updateTimer);
                }

                // Set a new timer to update prediction after 500ms of no slider movement
                if (currentMarker) {
                    updateTimer = setTimeout(() => {
                        updatePrediction(currentMarker.getLatLng());
                    }, 500);
                }
            });
        });

        // Handle map clicks
        map.on('click', function(e) {
            if (currentMarker) {
                map.removeLayer(currentMarker);
            }
            currentMarker = L.marker(e.latlng);
            currentMarker.addTo(map);
            updatePrediction(e.latlng);
        });

        async function updatePrediction(latlng) {
            try {
                // Show loading indicator
                const loadingElement = document.getElementById('loading');
                loadingElement.classList.add('show');

                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        ...features,
                        lat: latlng.lat,
                        lon: latlng.lng
                    })
                });

                // Hide loading indicator
                loadingElement.classList.remove('show');

                if (!response.ok) {
                    throw new Error('Server returned an error');
                }

                const data = await response.json();
                if (data.error) {
                    throw new Error(data.error);
                }

                const severity = data.severity;
                const severityText = ['Low', 'Medium', 'High', 'Very High'][severity - 1];
                const severityClass = severity > 2 ? 'severity-high' : severity === 2 ? 'severity-medium' : 'severity-low';

                // Update Highway Flag based on road type
                updateRoadTypeDisplay(data.road_type, data.road_name);

                // Update prediction display
                document.getElementById('prediction').innerHTML = `
                    <h3>Current Prediction</h3>
                    <p>Location: ${latlng.lat.toFixed(4)}, ${latlng.lng.toFixed(4)}</p>
                    <p>Road Type: <span class="${data.road_type === 'Highway' ? 'highway' : data.road_type === 'Local Road' ? 'local-road' : 'unknown-road'}">${data.road_type}</span></p>
                    <p>Road Name: ${data.road_name}</p>
                    <p>Severity: <span class="${severityClass}">${severityText}</span></p>
                `;

                const popupContent = `
                    <strong>Road Type:</strong> ${data.road_type}<br>
                    <strong>Road Name:</strong> ${data.road_name}<br>
                    <strong>Predicted Severity:</strong> ${severityText}
                `;
                currentMarker.bindPopup(popupContent).openPopup();
            } catch (error) {
                // Hide loading indicator on error
                document.getElementById('loading').classList.remove('show');
                console.error('Error:', error);
                showError(`Failed to get prediction: ${error.message}`);
                document.getElementById('prediction').innerHTML = `
                    <h3>Error</h3>
                    <p>Failed to get prediction. Please try again.</p>
                `;
            }
        }
    </script>
</body>
</html>
"""

# Save the map with the HTML template
with open("accident_prediction_map.html", "w") as f:
    f.write(html_template)

if __name__ == '__main__':
    print("Starting Flask server...")
    port = find_available_port()
    if port:
        print(f"Server will be available at http://127.0.0.1:{port}")
        print("Opening browser automatically...")
        Timer(1.5, open_browser, args=(port,)).start()
        try:
            app.run(host='127.0.0.1', port=port, debug=False)
        except Exception as e:
            print(f"Error starting server: {e}")
    else:
        print("No available port found. Please try again later.")
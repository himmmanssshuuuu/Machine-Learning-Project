from flask import Flask,request, render_template
import numpy as np
import pickle
import pandas as pd

# Define valid crops and areas directly to avoid dataset dependency
valid_crops = ['Maize', 'Potatoes', 'Rice, paddy', 'Sorghum', 'Soybeans', 'Wheat', 'Coffee, green', 'Cassava',
               'Sweet potatoes', 'Yams', 'Sugar cane', 'Plantains and others']
valid_areas = [
    'Afghanistan', 'Albania', 'Algeria', 'Andorra', 'Angola', 'Antigua and Barbuda', 'Argentina', 'Armenia', 'Australia', 'Austria',
    'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bhutan',
    'Bolivia', 'Bosnia and Herzegovina', 'Botswana', 'Brazil', 'Brunei', 'Bulgaria', 'Burkina Faso', 'Burundi', 'Cambodia', 'Cameroon',
    'Canada', 'Cape Verde', 'Central African Republic', 'Chad', 'Chile', 'China', 'Colombia', 'Comoros', 'Congo', 'Costa Rica',
    'Croatia', 'Cuba', 'Cyprus', 'Czech Republic', 'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic', 'DR Congo', 'Ecuador',
    'Egypt', 'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia', 'Eswatini', 'Ethiopia', 'Fiji', 'Finland', 'France',
    'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Greece', 'Grenada', 'Guatemala', 'Guinea', 'Guinea-Bissau',
    'Guyana', 'Haiti', 'Honduras', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq', 'Ireland',
    'Israel', 'Italy', 'Ivory Coast', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kiribati', 'Kuwait',
    'Kyrgyzstan', 'Laos', 'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Liechtenstein', 'Lithuania', 'Luxembourg',
    'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta', 'Marshall Islands', 'Mauritania', 'Mauritius', 'Mexico',
    'Micronesia', 'Moldova', 'Monaco', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia', 'Nauru',
    'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'North Korea', 'North Macedonia', 'Norway', 'Oman',
    'Pakistan', 'Palau', 'Palestine', 'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal',
    'Qatar', 'Romania', 'Russia', 'Rwanda', 'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Vincent and the Grenadines', 'Samoa', 'San Marino', 'Sao Tome and Principe',
    'Saudi Arabia', 'Senegal', 'Serbia', 'Seychelles', 'Sierra Leone', 'Singapore', 'Slovakia', 'Slovenia', 'Solomon Islands', 'Somalia',
    'South Africa', 'South Korea', 'South Sudan', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Sweden', 'Switzerland', 'Syria',
    'Taiwan', 'Tajikistan', 'Tanzania', 'Thailand', 'Timor-Leste', 'Togo', 'Tonga', 'Trinidad and Tobago', 'Tunisia', 'Turkey',
    'Turkmenistan', 'Tuvalu', 'Uganda', 'Ukraine', 'United Arab Emirates', 'United Kingdom', 'United States', 'Uruguay', 'Uzbekistan', 'Vanuatu',
    'Vatican City', 'Venezuela', 'Vietnam', 'Yemen', 'Zambia', 'Zimbabwe'
]

#flask app
app = Flask(__name__)

# Initialize models as None first
dtr = None
preprocessor = None

try:
    #loading models
    with open('models/dtr.pkl', 'rb') as f:
        dtr = pickle.load(f)
    with open('models/preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
except Exception as e:
    print(f"Error loading models: {str(e)}")
    # Set up dummy models for testing
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.preprocessing import StandardScaler
    dtr = DecisionTreeRegressor()
    preprocessor = StandardScaler()

@app.route('/')
def index():
    print("Debug - Rendering index page")
    print(f"Debug - Number of crops: {len(valid_crops)}")
    print(f"Debug - Number of areas: {len(valid_areas)}")
    return render_template('index.html', crops=valid_crops, areas=valid_areas)

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form data
            Year = float(request.form['Year'])
            average_rain_fall_mm_per_year = float(request.form['average_rain_fall_mm_per_year'])
            pesticides_tonnes = float(request.form['pesticides_tonnes'])
            avg_temp = float(request.form['avg_temp'])
            Area = request.form['Area']
            Item = request.form['Item']

            print(f"Debug - Input values: Year={Year}, Rain={average_rain_fall_mm_per_year}, Pest={pesticides_tonnes}, Temp={avg_temp}, Area={Area}, Item={Item}")

            # Validate country and crop
            if Area not in valid_areas:
                return render_template('index.html', 
                                    error="Please select a valid country",
                                    crops=valid_crops,
                                    areas=valid_areas)
            
            if Item not in valid_crops:
                return render_template('index.html',
                                    error="Please select a valid crop",
                                    crops=valid_crops,
                                    areas=valid_areas)

            # Make prediction
            features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)
            
            if dtr is None or preprocessor is None:
                prediction_value = 5000.0  # Dummy value for testing
            else:
                transformed_features = preprocessor.transform(features)
                prediction = dtr.predict(transformed_features)
                prediction_value = float(prediction[0])
            
            print(f"Debug - Prediction value: {prediction_value}")

            return render_template('index.html',
                                prediction=[prediction_value],
                                crops=valid_crops,
                                areas=valid_areas,
                                input_data={
                                    'Year': Year,
                                    'rainfall': average_rain_fall_mm_per_year,
                                    'pesticides': pesticides_tonnes,
                                    'temperature': avg_temp,
                                    'area': Area,
                                    'item': Item
                                })
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return render_template('index.html', prediction=np.array([[0.0]]), crops=valid_crops, areas=valid_areas)

if __name__=="__main__":
    app.run(debug=True, port=8080)
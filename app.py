from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and individual label encoders
model = joblib.load('models/car_price_predictor.pkl')
le_fuel = joblib.load('models/le_fuel.pkl')
le_seller = joblib.load('models/le_seller.pkl')
le_transmission = joblib.load('models/le_transmission.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = request.form.to_dict()

    try:
        # Preprocess input data
        input_data = pd.DataFrame([[
            float(data['Present_Price']),
            int(data['Kms_Driven']),
            le_fuel.transform([data['Fuel_Type']])[0] if data['Fuel_Type'] in le_fuel.classes_ else -1,
            le_seller.transform([data['Seller_Type']])[0] if data['Seller_Type'] in le_seller.classes_ else -1,
            le_transmission.transform([data['Transmission']])[0] if data['Transmission'] in le_transmission.classes_ else -1,
            int(data['Owner']),
            2024 - int(data['Year'])  # Calculate Age
        ]], columns=['Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner', 'Age'])

        # If any category wasn't found in the encoder, return an error message
        if -1 in input_data.values:
            return render_template('index.html', prediction="Error: Invalid category entered.")

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Render result
        return render_template('index.html', prediction=f'Predicted Price: ₹{prediction:.2f} Lakhs')

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)

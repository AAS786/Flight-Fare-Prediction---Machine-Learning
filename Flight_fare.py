import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load the model
loaded_model = pickle.load(open('flight_fare.sav', 'rb'))

# Load the dataset
df = pd.read_csv('Clean_Dataset.csv')

# Get unique values for categorical columns
airlines = df['airline'].unique().tolist()
flights = df['flight'].unique().tolist()  # Flight numbers
source_cities = df['source_city'].unique().tolist()
destination_cities = df['destination_city'].unique().tolist()
classes = df['class'].unique().tolist()

# Categorical to numeric mapping (based on the model training data)
airline_mapping = {v: i for i, v in enumerate(airlines)}
flight_mapping = {v: i for i, v in enumerate(flights)}  # Add mapping for flight numbers
source_city_mapping = {v: i for i, v in enumerate(source_cities)}
destination_city_mapping = {v: i for i, v in enumerate(destination_cities)}
class_mapping = {v: i for i, v in enumerate(classes)}
stops_mapping = {'zero': 0, 'one': 1, 'two_or_more': 2}
time_mapping = {'Early_Morning': 0, 'Morning': 1, 'Afternoon': 2, 'Evening': 3, 'Night': 4}

# Define the prediction function
def check(input_data):
    array_input = np.array(input_data)
    reshaped_input = array_input.reshape(1, -1)
    prediction = loaded_model.predict(reshaped_input)
    return format(prediction[0])

# Main function to render the Streamlit app
def main():
    # Custom CSS and font styling
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Pacifico&display=swap');
        
        body {
            background-image: url("https://i.postimg.cc/LsgBmkxf/Plane.jpg");
            background-size: cover;
            background-position: center;
        }

        .custom-title {
            font-family: 'Pacifico', cursive;
            font-size: 50px;
            color: #FF4B4B;
            text-align: center;
            margin-bottom: 30px;
        }

        .stApp {
            background: rgba(255, 255, 255, 0.85); /* Optional white overlay to increase contrast */
            border-radius: 10px;
            padding: 10px;
        }

        .stButton button {
            background-color: #4CAF50;
            color: white;
            padding: 1px 40px;
            text-align: center;
            font-size: 20px;
            margin: 8px 2px;
            width: 100%;
            border-radius: 24px;
            transition-duration: 0.4s;
        }

        .stButton button:hover {
            background-color: #45a049;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Custom title with font styling
    st.markdown('<h1 class="custom-title">✈️ Flight Fare Prediction</h1>', unsafe_allow_html=True)
    st.write("Please fill out the details below to get an accurate prediction of your flight fare!")

    # Create columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        airline = st.selectbox("✈️ **Airline**", options=airlines)
        flight = st.selectbox("🛫 **Flight Number**", options=flights)  # Select flight from options
        source_city = st.selectbox("🏙️ **Source City**", options=source_cities)
        departure_time = st.selectbox("⏰ **Departure Time**", options=['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night'])

    with col2:
        stops = st.selectbox("🛑 **Stops**", options=['zero', 'one', 'two_or_more'])
        arrival_time = st.selectbox("🕑 **Arrival Time**", options=['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night'])
        destination_city = st.selectbox("🌆 **Destination City**", options=destination_cities)
        travel_class = st.selectbox("🎟️ **Class**", options=classes)
    
    # Place duration and days left below
    col3, col4 = st.columns(2)
    
    with col3:
        duration = st.number_input("⏳ **Duration (hours)**", min_value=0.0, step=0.1)
    
    with col4:
        days_left = st.number_input("📅 **Days Left**", min_value=1, step=1)

    # Encode categorical variables to match model input format
    airline_encoded = airline_mapping[airline]
    flight_encoded = flight_mapping[flight]  # Encode the flight number
    source_city_encoded = source_city_mapping[source_city]
    destination_city_encoded = destination_city_mapping[destination_city]
    travel_class_encoded = class_mapping[travel_class]
    stops_encoded = stops_mapping[stops]
    departure_time_encoded = time_mapping[departure_time]
    arrival_time_encoded = time_mapping[arrival_time]

    # Center the prediction button
    pred = ""
    col_center = st.columns([1, 1, 1])  # Adjust button position
    with col_center[1]:
        if st.button("🚀 Click Here for Flight Fare Prediction"):
            pred = check([airline_encoded, flight_encoded, source_city_encoded, departure_time_encoded, stops_encoded, arrival_time_encoded, destination_city_encoded, travel_class_encoded, duration, days_left])
            st.balloons()  # Show balloons when the prediction is made
            st.success(f"Your Flight Fare is ₹{pred}")

if __name__ == '__main__':
    main()

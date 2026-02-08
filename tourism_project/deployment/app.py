
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="indianakhil/tourism-package-prediction-model", filename="best_tourism_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Prediction
st.title("Tourism Package Purchase Prediction App")
st.write("The Tourism Package Purchase Prediction App is an internal tool for travel company staff that predicts whether customers are likely to purchase the Wellness Tourism Package based on their details.")
st.write("Kindly enter the customer details to check whether they are likely to purchase the package.")

# Collect user input
Age = st.number_input("Age (customer's age in years)", min_value=18, max_value=100, value=35)
TypeofContact = st.selectbox("Type of Contact (how the customer was contacted)", ["Company Invited", "Self Enquiry"])
CityTier = st.selectbox("City Tier (city category based on development)", [1, 2, 3])
DurationOfPitch = st.number_input("Duration of Pitch (in minutes)", min_value=0.0, max_value=60.0, value=15.0)
Occupation = st.selectbox("Occupation (customer's occupation)", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
Gender = st.selectbox("Gender (customer's gender)", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=0.0, max_value=10.0, value=3.0)
ProductPitched = st.selectbox("Product Pitched (type of product pitched)", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
PreferredPropertyStar = st.selectbox("Preferred Property Star (preferred hotel rating)", [3.0, 4.0, 5.0])
MaritalStatus = st.selectbox("Marital Status (customer's marital status)", ["Single", "Married", "Divorced", "Unmarried"])
NumberOfTrips = st.number_input("Number of Trips (average trips per year)", min_value=0.0, max_value=20.0, value=2.0)
Passport = st.selectbox("Has Passport?", ["Yes", "No"])
PitchSatisfactionScore = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
OwnCar = st.selectbox("Owns Car?", ["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting (below age 5)", min_value=0, max_value=5, value=0)
Designation = st.selectbox("Designation (customer's designation)", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
MonthlyIncome = st.number_input("Monthly Income (gross monthly income)", min_value=0.0, value=20000.0)

# Encoding mappings (based on training data)
type_of_contact_map = {'Company Invited': 0, 'Self Enquiry': 1}
occupation_map = {'Salaried': 0, 'Small Business': 1, 'Large Business': 2, 'Free Lancer': 3}
gender_map = {'Male': 0, 'Female': 1}
product_pitched_map = {'Basic': 0, 'Standard': 1, 'Deluxe': 2, 'Super Deluxe': 3, 'King': 4}
marital_status_map = {'Single': 0, 'Married': 1, 'Divorced': 2, 'Unmarried': 3}
designation_map = {'Executive': 0, 'Manager': 1, 'Senior Manager': 2, 'AVP': 3, 'VP': 4}

# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': type_of_contact_map[TypeofContact],
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Occupation': occupation_map[Occupation],
    'Gender': gender_map[Gender],
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'ProductPitched': product_pitched_map[ProductPitched],
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': marital_status_map[MaritalStatus],
    'NumberOfTrips': NumberOfTrips,
    'Passport': 1 if Passport == "Yes" else 0,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': designation_map[Designation],
    'MonthlyIncome': MonthlyIncome
}])

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    result = "purchase the package" if prediction == 1 else "not purchase the package"
    st.write(f"Based on the information provided, the customer is likely to {result}.")

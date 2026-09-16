import streamlit as st
import pickle
import pandas as pd

# --------------------------------------------
# Load the trained model and scaler from files
# --------------------------------------------
def load_model_and_scaler(model_filename='iris_model.pkl', scaler_filename='iris_scaler.pkl'):
    """Loads the trained model and scaler from pickle files."""
    with open(model_filename, 'rb') as model_file:
        model = pickle.load(model_file)
    with open(scaler_filename, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

# --------------------------------------------
# Preprocess input data using the loaded scaler
# --------------------------------------------
def preprocess_input(input_data, scaler):
    """Preprocess the input data: convert to DataFrame and apply scaling."""
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    return input_scaled

# --------------------------------------------
# Predict the Iris flower species
# --------------------------------------------
def predict_iris_species(input_data):
    """Predicts the species of the Iris flower (Setosa, Versicolor, or Virginica)."""
    model, scaler = load_model_and_scaler()
    input_scaled = preprocess_input(input_data, scaler)
    prediction = model.predict(input_scaled)
    species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    predicted_species = species_map[prediction[0]]
    return predicted_species

# --------------------------------------------
# Streamlit Web App Interface
# --------------------------------------------
def main():
    st.title("🌸 Iris Species Prediction App")
    st.write("Enter the flower measurements below to predict its species:")

    # User input fields for the flower's measurements
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, value=5.1)
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, value=3.5)
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, value=1.4)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, value=0.2)

    # Button to make prediction
    if st.button("Predict"):
        input_data = {
            'sepal length (cm)': sepal_length,
            'sepal width (cm)': sepal_width,
            'petal length (cm)': petal_length,
            'petal width (cm)': petal_width
        }
        result = predict_iris_species(input_data)
        st.success(f"The predicted Iris species is: **{result.capitalize()}**")

if __name__ == "__main__":
    main()

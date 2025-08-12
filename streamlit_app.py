import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Load the trained model and preprocessing pipeline from the pickle file
try:
    loaded_objects = joblib.load('model_pipeline.pkl')
    model = loaded_objects['model']
    pipeline = loaded_objects['pipeline'] # This should be the StandardScaler pipeline
    feature_names = loaded_objects['feature_names'] # List of numerical feature names
    st.success("Model and preprocessing pipeline loaded successfully!")
except FileNotFoundError:
    st.error("Error: 'model_pipeline.pkl' not found. Please make sure the file is in the same directory.")
    st.stop()
except KeyError:
    st.error("Error: 'model_pipeline.pkl' does not contain expected objects. Please check the file content.")
    st.stop()


st.title('Song Popularity Predictor')

st.write("""
This app predicts the popularity of a song based on its audio features.
""")

# Input features from the user
st.sidebar.header('Song Features')

def user_input_features():
    acousticness = st.sidebar.slider('Acousticness', 0.0, 1.0, 0.5)
    danceability = st.sidebar.slider('Danceability', 0.0, 1.0, 0.5)
    energy = st.sidebar.slider('Energy', 0.0, 1.0, 0.5)
    instrumentalness = st.sidebar.slider('Instrumentalness', 0.0, 1.0, 0.5)
    key = st.sidebar.slider('Key', 0, 11, 5)
    liveness = st.sidebar.slider('Liveness', 0.0, 1.0, 0.5)
    loudness = st.sidebar.slider('Loudness', -60.0, 0.0, -30.0)
    audio_mode_label = st.sidebar.radio('Audio Mode', ['Minor', 'Major'])
    # Mengubah label kembali ke nilai numerik (0 atau 1)
    if audio_mode_label == 'Major':
        audio_mode = 1
    else:
        audio_mode = 0
    #audio_mode = st.sidebar.radio('Audio Mode', [0, 1])
    speechiness = st.sidebar.slider('Speechiness', 0.0, 1.0, 0.5)
    tempo = st.sidebar.slider('Tempo', 0.0, 250.0, 120.0)
    time_signature = st.sidebar.slider('Time Signature', 0, 5, 4)
    audio_valence = st.sidebar.slider('Audio Valence', 0.0, 1.0, 0.5)
    song_duration_ms = st.sidebar.slider('Song Duration (ms)', 0, 1000000, 200000)


    data = {'Song duration': song_duration_ms,
            'Acousticness': acousticness,
            'Danceability': danceability,
            'Energy': energy,
            'instrumentalness': instrumentalness,
            'Key': key,
            'Liveness': liveness,
            'Loudness': loudness,
            'Audio mode': audio_mode,
            'Speechiness': speechiness,
            'Tempo': tempo,
            'Time signature': time_signature,
            'Audio valence': audio_valence}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Ensure the order of columns matches the training data and convert to numpy array
# Explicitly reindex the input_df to match the order of feature_names
# The 'song_name' column is not present in input_df from user_input_features function, which is correct.
input_data_ordered = input_df.reindex(columns=feature_names) # Keep as DataFrame for pipeline

# Display the input features
st.subheader('User Input Features')
st.write(input_df)

# Make prediction
if st.sidebar.button('Predict Popularity'):
    try:
        # Apply the loaded preprocessing pipeline (StandardScaler) to the input data
        processed_input = pipeline.transform(input_data_ordered)

        # Make prediction using the loaded model
        prediction = model.predict(processed_input)
        st.subheader('Predicted Song Popularity')
        # Assuming popularity is on a scale of 0-100
        st.write(f"{prediction[0]:.2f}")
        st.write(f"Note: Score 0-100")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

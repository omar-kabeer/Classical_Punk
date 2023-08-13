import streamlit as st
import numpy as np
import joblib
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import librosa
model = svm.SVC(kernel='rbf', C=10, gamma=0.1, probability=True)
scaler = StandardScaler()
pca = PCA(n_components=20)

model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')
pca = joblib.load('pca.joblib')

def app():
    st.title('Music Genre Classifier')
    st.write('Upload an audio file and the classifier will predict the musical genre.')

    # Upload Audio File
    audio_file = st.file_uploader('Upload Audio File', type=['wav', 'mp3'])

    if audio_file is not None:
        # Load Audio Data
        audio_data, sample_rate = librosa.load(audio_file, res_type='kaiser_fast')

        # Extract Features
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = scaler.transform(mfccs)
        mfccs_pca = pca.transform(mfccs_scaled)
        features = mfccs_pca.reshape(1, -1)

        # Predict Genre
        genres = ['Classical', 'Electronic', 'Folk/Country', 'Hip-Hop', 'Jazz', 'Pop', 'Rock']
        prediction = model.predict(features)[0]
        probability = np.max(model.predict_proba(features))
        st.write(f'Prediction: {genres[prediction]} (Probability: {probability:.2f})')
if __name__ == '__main__':
    app()
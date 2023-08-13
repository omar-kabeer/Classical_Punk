import os
import librosa
import numpy as np

# Function to extract features from an audio file
def extract_features(audio_file_path):
    features = []

    # Load audio
    audio, sr = librosa.load(audio_file_path, sr=None)

    # Extract features
    mfccs = librosa.feature.mfcc(audio, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(audio, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(audio, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(audio, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(audio, sr=sr)
    spectral_flatness = librosa.feature.spectral_flatness(audio)
    harmonicity = librosa.effects.harmonic(audio)
    percussiveness = librosa.effects.percussive(audio)
    spectral_flux = librosa.onset.onset_strength(audio, sr=sr)
    spectral_slope = librosa.feature.spectral_slope(audio, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(audio)
    energy = np.sum(audio ** 2)
    tempo, _ = librosa.beat.beat_track(audio, sr=sr)
    pitch, _ = librosa.piptrack(audio, sr=sr)
    loudness = librosa.amplitude_to_db(librosa.feature.rms(audio))
    duration = librosa.get_duration(y=audio, sr=sr)
    complexity = np.mean(np.diff(audio))
    timbre = np.mean(spectral_contrast)
    spectral_energy = np.sum(np.abs(np.fft.fft(audio)) ** 2)
    temporal_energy = np.sum(audio ** 2)
    energy_flux = np.diff(temporal_energy)
    energy_distribution = temporal_energy / np.sum(temporal_energy)
    energy_entropy = -np.sum(energy_distribution * np.log2(energy_distribution + 1e-12))

    # Append features to the list
    features.extend([
        mfccs,
        spectral_centroid,
        spectral_bandwidth,
        spectral_rolloff,
        spectral_contrast,
        spectral_flatness,
        harmonicity,
        percussiveness,
        spectral_flux,
        spectral_slope,
        zcr,
        energy,
        tempo,
        pitch,
        loudness,
        duration,
        complexity,
        timbre,
        spectral_energy,
        temporal_energy,
        energy_flux,
        energy_distribution,
        energy_entropy,
    ])

    return features


# List to store extracted features
all_features = []

# Loop through audio files and extract features
for genre in genres:
    genre_dir = f'../data/genres/{genre}'
    for filename in os.listdir(genre_dir):
        if filename.endswith(".wav"):
            audio_path = os.path.join(genre_dir, filename)
            features = extract_features(audio_path)
            all_features.append(features)

# Convert the list of feature sets into a DataFrame
feature_columns = [
    "mfccs", "spectral_centroid", "spectral_bandwidth", "spectral_rolloff", 
    "spectral_contrast", "spectral_flatness", "harmonicity", "percussiveness", 
    "spectral_flux", "spectral_slope", "zcr", "energy", "tempo", "pitch", 
    "loudness", "duration", "complexity", "timbre", "spectral_energy", 
    "temporal_energy", "energy_flux", "energy_distribution", "energy_entropy"
]

feature_df = pd.DataFrame(all_features, columns=feature_columns)

# Save the DataFrame as a CSV file
csv_filename = "audio_features.csv"
feature_df.to_csv(csv_filename, index=False)
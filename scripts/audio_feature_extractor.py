import os
import librosa
import numpy as np
import pandas as pd
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
data = []
for genre in genres:
    genre_dir = f'../data/genres/{genre}'
    for filename in os.listdir(genre_dir):
        if filename.endswith('.wav'):
            audio_path = os.path.join(genre_dir, filename)
            y, sr = librosa.load(audio_path)

            # Extract features
            central_moments = np.asarray([np.mean((y - np.mean(y)) ** i) for i in range(1, 5)])
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
            rmse = librosa.feature.rms(y=y)
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            spectral_flatness = librosa.feature.spectral_flatness(y=y)
            loudness = librosa.amplitude_to_db(librosa.feature.rms(y=y))
            complexity = np.mean(np.diff(y))
            timbre = np.mean(spectral_contrast)
            spectral_energy = np.sum(np.abs(np.fft.fft(y)) ** 2)
            temporal_energy = np.sum(y ** 2)
            harmonicity = librosa.effects.harmonic(y=y)
            percussiveness = librosa.effects.percussive(y=y)

            # Add features to dataframe
            features = [genre, filename, beats, sr, central_moments, zero_crossing_rate[0], rmse[0],
                        tempo, spectral_contrast, spectral_rolloff, mfccs, chroma, spectral_centroid, spectral_bandwidth,
                        spectral_flatness, loudness, complexity, timbre, spectral_energy, temporal_energy, harmonicity, percussiveness]
            data.append(features)

df = pd.DataFrame(data, columns=['Genre', 'Filename', 'Beats', 'SR', 'Central Moments', 'Zero Crossing Rate', 'RMSE', 'Tempo', 'Spectral Contrast',
                                'Spectral Roll-off', 'MFCC', 'Chroma', 'Spectral Centroid', 'Spectral Bandwidth',
                                'Spectral Flatness', 'Loudness', 'Complexity', 'Timbre', 'Spectral Energy', 'Temporal Energy', 'Harmonicity', 'Percussiveness'])

# Save the DataFrame as a CSV file
csv_filename = "../data/data.csv"
df.to_csv(csv_filename, index=False)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import librosa

def count_values_above_mean(values):
    """
    Count the number of values in a list that are above the mean.

    Args:
        values (list): The list of values.

    Returns:
        int: The count of values above the mean.
    """
    # Calculate the mean of the values
    mean = sum(values) / len(values)

    # Count the number of values that are above the mean
    count = sum(1 for value in values if value > mean)
    
    return count

# Function to count values below the mean
def count_values_below_mean(values):
    """
    Calculates the number of values in a list that are below the mean.

    Parameters:
        values (list): A list of numerical values.

    Returns:
        int: The count of values below the mean.
    """
    mean = sum(values) / len(values)
    return sum(1 for value in values if value < mean)


def show_waveform(x, num):
    """
    Display a waveform plot of a WAV file.

    Parameters:
    - x (str): The genre of the WAV file.
    - num (int): The number of the WAV file.

    Returns:
    - None
    """
    # Load WAV file
    wav_file = f'../data/genres/{x}/{x}.{num}.wav'
    y, sr = librosa.load(wav_file)
# Create x-axis values
    time = librosa.times_like(y, sr=sr)

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 8))
    sns.lineplot(x=time, y=y)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Sample waveform for {x}')
    plt.show()

def show_spectogram(x, num):
    """
    Show the spectrogram for a given audio file.

    Parameters:
        x (str): The genre of the audio file.
        num (int): The number of the audio file.

    Returns:
        None
    """
    # Load audio file
    audio_path = f'../data/genres/{x}/{x}.{num}.wav'
    y, sr = librosa.load(audio_path)

    # Calculate spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

    # Convert to decibels
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Create figure
    plt.figure(figsize=(10, 4))
    ax = sns.heatmap(S_dB, cmap='viridis')

    # Set x and y axis labels
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency (Hz)')

    # Set figure title
    ax.set_title(f'Sample spectrogram for {x}')

    # Show figure
    plt.show()
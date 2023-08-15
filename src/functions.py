import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import os
from scipy.stats import skew, kurtosis

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
    
def show_sr(x, num):
    """
    Generate a plot of the spectral rolloff for a given audio file.

    Parameters:
    - x (str): The genre of the audio file.
    - num (int): The number of the audio file.

    Returns:
    - None

    This function loads an audio file based on the given genre and number. It then computes
    the spectral rolloff of the audio file and plots it using matplotlib. The plot shows the
    spectral rolloff over time, with the x-axis representing the frame and the y-axis 
    representing the frequency in Hz.

    Example usage:
    show_sr('rock', 1)
    """
    # Load audio file
    audio_path = f'../data/genres/{x}/{x}.{num}.wav'
    y, sr = librosa.load(audio_path)

    # Compute spectral rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]

    # Create plot
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    plt.plot(spectral_rolloff, color='blue')
    plt.title(f'Sample spectral rolloff for {x}')
    plt.xlabel('Frame')
    plt.ylabel('Frequency (Hz)')
    plt.show()


def show_chroma(genre, num):
    """
    Generates a heatmap of the chroma feature for a given audio file.

    Parameters:
    - genre (str): The genre of the audio file.
    - num (int): The number of the audio file.

    Returns:
    - None
    """
    # Define audio file path
    audio_path = os.path.join('../data/genres', genre, f'{genre}.{num}.wav')

    # Load audio file
    y, sr = librosa.load(audio_path)

    # Compute chroma feature
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    # Create time axis in seconds
    time = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr)

    # Create chroma note names
    chroma_note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    # Create dataframe
    df = pd.DataFrame(chroma, index=chroma_note_names, columns=time)

    # Plot heatmap using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, cmap='viridis', xticklabels=50, yticklabels=1)
    plt.title(f'Sample chroma feature for {genre}')
    plt.xlabel('Time (s)')
    plt.ylabel('Chroma Note')
    plt.show()


def show_zcr(x, num):
    """
    Generate a plot of the zero crossing rate (ZCR) for an audio file.

    Parameters:
    - x (str): The genre of the audio file.
    - num (int): The number of the audio file.

    Returns:
    None
    """
    # Load audio file
    audio_path = os.path.join('..', 'data', 'genres', x, f'{x}.{num}.wav')
    y, sr = librosa.load(audio_path)

    # Compute zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)

    # Plot with Seaborn
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=zcr[0], ax=ax)
    ax.set(title=f'Zero Crossing Rate for {x} Genre', xlabel='Time (s)', ylabel='ZCR')
    plt.show()


def skew_kurt(data, col):
    """
    Calculate the skewness and kurtosis of a specified column in a given dataset.
    
    Parameters:
        data (pandas.DataFrame): The dataset containing the column.
        col (str): The name of the column to calculate the skewness and kurtosis for.
        
    Returns:
        None
    """
    # Calculate skewness and kurtosis of specified column
    skewness = skew(data[col])
    kurtosis = kurtosis(data[col])

    # Create histogram of specified column with mean, median, and mode
    sns.histplot(data=data, x=col, kde=True)
    plt.axvline(skewness, color='r', linestyle='--', label='Mean')
    plt.axvline(kurtosis, color='g', linestyle='--', label='Median')
    plt.axvline(data[col].mode()[0], color='b', linestyle='--', label='Mode')
    plt.legend()

    # Add text annotation for skewness and kurtosis values
    plt.annotate(f'Skewness: {skewness:.2f}', xy=(0.5, 0.9), xycoords='axes fraction')
    plt.annotate(f'Kurtosis: {kurtosis:.2f}', xy=(0.5, 0.85), xycoords='axes fraction')

    plt.show()

def modify(value):
    """
    Modify the given value by adding a leading '0' if it starts with a '.'
    and adding a trailing '0' if it ends with a '.'.

    Parameters:
    - value: a string representing the value to be modified.

    Returns:
    - The modified value as a string.
    """
    if value.startswith('.'):
        value = '0' + value
    if value.endswith('.'):
        value = value + '0'
    return value

# Function to process the string and extract float values
def process_list(value):
    """
    Process a list of strings and convert it into a list of floats.

    Parameters:
    - value (list): A list of strings.

    Returns:
    - cleaned_values (list): A list of floats after cleaning the values.
    """
    value = "".join(value)
    value = value.replace('[', '').replace(']', '')
    values = value.split()  # Split the string into a list of values
    cleaned_values = [float(val) for val in values if val != '...']  # Convert to float, excluding '...'
    return cleaned_values
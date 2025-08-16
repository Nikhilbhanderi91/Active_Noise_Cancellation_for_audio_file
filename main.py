import os
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import matplotlib.pyplot as plt


def active_noise_cancellation(noise_signal, reference_signal, mu=0.01):
    """
    Implement Active Noise Cancellation using the LMS (Least Mean Squares) algorithm
    Parameters:
    - noise_signal: The noisy audio signal to be cleaned
    - reference_signal: A reference noise signal used to adaptively filter
    - mu: Step size (learning rate) for the adaptive filter
    
    Returns:
    - cleaned_signal: Noise-reduced audio signal
    - error_signal: Difference between input and estimated noise
    """
    
    noise_signal = np.array(noise_signal, dtype=float)
    reference_signal = np.array(reference_signal, dtype=float)

    # Ensure reference signal is long enough for the filter length
    filter_length = min(50, len(noise_signal) // 2)
    weights = np.zeros(filter_length)
    
    cleaned_signal = np.zeros_like(noise_signal, dtype=float)
    error_signal = np.zeros_like(noise_signal, dtype=float)
    
    for n in range(filter_length, len(noise_signal)):
        
        # Extract reference signal window
        x = reference_signal[n-filter_length:n]
        
        # Ensure that the window and weights have compatible shapes
        if x.shape[0] != weights.shape[0]:
            x = np.resize(x, weights.shape)
    
        estimated_noise = np.dot(x, weights)
        error_signal[n] = noise_signal[n] - estimated_noise
        weights += 2 * mu * error_signal[n] * x
        cleaned_signal[n] = noise_signal[n] - estimated_noise
    
    return cleaned_signal, error_signal

def analyze_noise_characteristics(audio_data, rate):
    """
    Analyze noise characteristics of the audio signal
    
    Parameters:
    - audio_data: Input audio data
    - rate: Sampling rate
    
    Returns:
    - Dictionary of noise characteristics
    """
    duration = len(audio_data) / rate
    mean_amplitude = np.mean(np.abs(audio_data))
    peak_amplitude = np.max(np.abs(audio_data))
    
    
    frequencies, power_spectrum = signal.periodogram(audio_data, rate)
    dominant_freq = frequencies[np.argmax(power_spectrum)]
    
    noise_profile = {
        'duration': duration,
        'mean_amplitude': mean_amplitude,
        'peak_amplitude': peak_amplitude,
        'dominant_frequency': dominant_freq,
        'recommendations': []
    }
    if duration < 2:
        noise_profile['recommendations'].append("Short audio clip detected. Consider more precise noise reduction.")
    
    if mean_amplitude < 0.1:
        noise_profile['recommendations'].append("Low signal level detected. Amplification may be needed.")
    return noise_profile

def plot_signals(original, cleaned, error):
    """
    Plot original, cleaned, and error signals
    Parameters:
    - original: Original noisy audio signal
    - cleaned: Noise-reduced audio signal
    - error: Error signal from noise cancellation
    """
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    plt.title('Original Noisy Signal')
    plt.plot(original)
    plt.ylabel('Amplitude')
    
    plt.subplot(3, 1, 2)
    plt.title('Cleaned Signal')
    plt.plot(cleaned)
    plt.ylabel('Amplitude')
    
    plt.subplot(3, 1, 3)
    plt.title('Error Signal')
    plt.plot(error)
    plt.ylabel('Amplitude')
    plt.xlabel('Sample')
    
    plt.tight_layout()
    plt.show()
    
def process_audio_noise_cancellation(input_file, output_file=None, plot=True):
    """
    Process audio file for noise cancellation
    
    Parameters:
    - input_file: Path to the input audio file
    - output_file: Path to save the noise-reduced audio file (optional)
    - plot: Whether to plot signal comparisons (default: True)
    
    Returns:
    - Noise profile dictionary
    """
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"The file {input_file} does not exist.")
    
    rate, data = wavfile.read(input_file)
    
    if len(data.shape) == 2:
        cleaned_channels = []
        error_channels = []

        
        for channel in range(data.shape[1]):
            reference_noise = data[:len(data)//4, channel]
            cleaned_channel, error_channel = active_noise_cancellation(
                noise_signal=data[:, channel], 
                reference_signal=reference_noise
            )
            cleaned_channels.append(cleaned_channel)
            error_channels.append(error_channel)
        
        cleaned_data = np.column_stack(cleaned_channels)
        
        if plot:
            plot_signals(data[:, 0], cleaned_channels[0], error_channels[0])
    else:
        reference_noise = data[:len(data)//4]
        cleaned_data, error_signal = active_noise_cancellation(
            noise_signal=data, 
            reference_signal=reference_noise
        )
        
        if plot:
            plot_signals(data, cleaned_data, error_signal)
    
    cleaned_data = cleaned_data / np.max(np.abs(cleaned_data))
    cleaned_data = (cleaned_data * 32767).astype(np.int16)
    
    if output_file is None:
        filename, ext = os.path.splitext(input_file)
        output_file = f"{filename}_noise_reduced{ext}"
    
    wavfile.write(output_file, rate, cleaned_data)
    
    noise_profile = analyze_noise_characteristics(data, rate)
    
    print("Noise Characteristics:")
    for key, value in noise_profile.items():
        print(f"{key}: {value}")
    
    return noise_profile

def main():
    # Directly set the input file path to /Users/nikhilbhanderi/Desktop/sample.wav
    input_file = '/Users/nikhilbhanderi/Desktop/sample.wav'
    
    try:
        process_audio_noise_cancellation(input_file)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()

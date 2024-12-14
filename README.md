# Active Noise Cancellation for Audio Files

This repository implements **Active Noise Cancellation (ANC)** for audio files using adaptive filtering with the Least Mean Squares (LMS) algorithm. The project reduces unwanted noise from audio recordings, providing cleaner sound through real-time noise reduction and customizable filtering.

## Features

- **Noise Reduction**: Removes background noise from audio using the LMS algorithm.
- **Multi-Channel Support**: Processes stereo or mono audio files.
- **Signal Analysis**: Analyzes noise characteristics such as duration, amplitude, and dominant frequencies.
- **Plotting**: Visualizes the original, cleaned, and error signals for easy comparison.
- **Customizable Parameters**: Allows tuning of learning rate (`mu`) for adaptive filtering.

## Technologies Used

- **Python**: Main implementation language.
- **NumPy**: For efficient numerical operations.
- **SciPy**: For signal processing and audio manipulation.
- **Matplotlib**: For plotting audio signals.
- **PyDub/Librosa**: (Optional) For advanced audio processing.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/active-noise-cancellation.git
    ```

2. Install dependencies:
    ```bash
    pip install numpy scipy matplotlib
    ```

## Usage

1. Place the input audio file in your desired location (e.g., `sample.wav`).
2. Run the script to process the audio and reduce noise:
    ```bash
    python noise_cancellation.py
    ```

3. Optionally, specify an output file or disable plotting by modifying the function call:
    ```python
    process_audio_noise_cancellation('path/to/input.wav', 'path/to/output.wav', plot=False)
    ```

## Example

In the `main()` function, the code processes the audio file stored at:

`/Users/nikhilbhanderi/Desktop/sample.wav`

Make sure to change the input file path if needed.

## Noise Profile Analysis

The script analyzes the noise profile of the audio, including:
- **Duration** of the audio
- **Mean** and **Peak Amplitude**
- **Dominant Frequency**

Example Output:

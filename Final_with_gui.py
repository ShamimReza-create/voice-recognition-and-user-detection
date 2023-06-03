import tkinter as tk
import numpy as np
import os
from scipy.spatial.distance import cosine
from scipy.io import wavfile
import sounddevice as sd
import matplotlib.pyplot as plt

# Set the path to the train directory
TRAIN_DIR = "DataSets/1"

# Set the sample rate and duration for audio recording
SAMPLE_RATE = 44100  # Adjust if necessary
DURATION = 5  # Adjust the duration for each audio sample

# Lists to store accuracy values
accuracy_values = []
iteration_values = []

# Function to extract audio features
def extract_features(audio_data):
    audio_data = audio_data.astype(float)
    return np.mean(audio_data), np.std(audio_data)

# Function to capture audio and save it as a WAV file
def capture_audio(label):
    print(f"Speak now for label '{label}'...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()

    filename = f"{label}.wav"
    filepath = os.path.join(TRAIN_DIR, filename)
    wavfile.write(filepath, SAMPLE_RATE, audio)

    print(f"Audio sample saved as '{filename}'.")

# Train the speaker recognition model
def train_model():
    train_features = []
    train_labels = []

    for filename in os.listdir(TRAIN_DIR):
        if filename.endswith(".wav"):
            audio_path = os.path.join(TRAIN_DIR, filename)
            _, audio_data = wavfile.read(audio_path)
            mean, std = extract_features(audio_data)
            train_features.append((mean, std))
            train_labels.append(filename.split('.')[0])

    return train_features, train_labels

# Recognize the user
def recognize_user(test_features, train_features, train_labels):
    recognized_users = []

    for test_feature in test_features:
        min_distance = float("inf")
        recognized_user = None

        for i, train_feature in enumerate(train_features):
            distance = cosine(test_feature, train_feature)
            if distance < min_distance:
                min_distance = distance
                recognized_user = train_labels[i]

        recognized_users.append(recognized_user)

    return recognized_users

# Function to handle the recognition process
def recognize():
    # Capture user voice input from the microphone
    print("Speak now...")
    voice_input = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()

    # Extract features from the voice input
    voice_features = extract_features(voice_input)

    # Train the speaker recognition model
    train_features, train_labels = train_model()

    # Recognize the user
    recognized_users = recognize_user([voice_features], train_features, train_labels)

    # Update the label in the GUI
    result_label.config(text=f"Recognized user: {recognized_users[0]}")

# Function to calculate the accuracy of the model
def calculate_accuracy():
    global accuracy_values, iteration_values

    test_features = []
    test_labels = []

    for filename in os.listdir(TRAIN_DIR):
        if filename.endswith(".wav"):
            audio_path = os.path.join(TRAIN_DIR, filename)
            _, audio_data = wavfile.read(audio_path)
            mean, std = extract_features(audio_data)
            test_features.append((mean, std))
            test_labels.append(filename.split('.')[0])

    train_features, train_labels = train_model()
    recognized_users = recognize_user(test_features, train_features, train_labels)

    correct_count = sum(1 for i in range(len(test_labels)) if test_labels[i] == recognized_users[i])
    accuracy = correct_count / len(test_labels)

    iteration_values.append(len(accuracy_values) + 1)
    accuracy_values.append(accuracy)

    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Update the accuracy graph
    update_accuracy_graph()

# Function to update the accuracy graph
def update_accuracy_graph():
    plt.figure(figsize=(8, 5))
    plt.plot(iteration_values, accuracy_values, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Function to handle the addition of new users
def add_user():
    # Get the label for the new user from the entry field
    label = entry.get()

    # Capture audio for the new user
    capture_audio(label)

    # Update the entry field
    entry.delete(0, tk.END)

# Create the GUI window
window = tk.Tk()
window.title("Speaker Recognition")

# Create a label for the result
result_label = tk.Label(window, text="Recognized user:")
result_label.pack(pady=10)

# Create an entry field for adding new users
entry = tk.Entry(window)
entry.pack(pady=5)

# Create a button to add new users
add_button = tk.Button(window, text="Add User", command=add_user)
add_button.pack(pady=5)

# Create a button to start the recognition process
recognize_button = tk.Button(window, text="Start Recognition", command=recognize)
recognize_button.pack(pady=10)

# Create a button to calculate the accuracy
accuracy_button = tk.Button(window, text="Calculate Accuracy", command=calculate_accuracy)
accuracy_button.pack(pady=10)

# Run the GUI main loop
window.mainloop()

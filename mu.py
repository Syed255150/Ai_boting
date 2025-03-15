import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess audio
def load_audio(file_path, sample_rate=16000):
    y, sr = librosa.load(file_path, sr=sample_rate, mono=True)
    return y, sr

# Extract pitch (fundamental frequency, F0)
def extract_pitch(y, sr):
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=80, fmax=1000, sr=sr)
    return f0

# Convert frequency (Hz) to MIDI notes
def hz_to_midi(f0):
    return librosa.hz_to_midi(f0)

# Detect note changes and measure durations
def detect_notes(f0, sr):
    midi_notes = hz_to_midi(f0)
    note_changes = np.where(np.diff(midi_notes) != 0)[0]  # Detect changes in MIDI notes
    note_start_times = librosa.frames_to_time(note_changes, sr=sr)

    # Calculate note durations
    note_durations = np.diff(note_start_times, append=note_start_times[-1])
    return note_start_times, midi_notes[note_changes], note_durations

# Save detected notes and durations to a file
def save_notes_to_file(note_start_times, midi_notes, note_durations, output_file="detected_notes.txt"):
    with open(output_file, "w") as f:
        for i in range(len(note_start_times)):
            f.write(f"Note {i+1}: MIDI {midi_notes[i]} (Start: {note_start_times[i]:.2f}s, Duration: {note_durations[i]:.2f}s)\n")
    print(f"Notes saved to {output_file}")

# Plot pitch over time
def plot_pitch(f0, sr):
    times = librosa.times_like(f0, sr=sr)
    plt.figure(figsize=(10, 4))
    plt.plot(times, f0, label="Pitch (Hz)", color="blue")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Pitch Estimation")
    plt.legend()
    plt.show()

# Main function
def main():
    audio_path = "C:/Users/abc/Desktop/music/Samjho Na X Wishes - Aditya Rikhari ft. Talwinder & King ｜ Tu Aake Dekh Le ｜ Chillout Vibes.wav"  # Change to your audio file
    y, sr = load_audio(audio_path)
    f0 = extract_pitch(y, sr)
    
    note_start_times, midi_notes, note_durations = detect_notes(f0, sr)
    save_notes_to_file(note_start_times, midi_notes, note_durations)
    
    plot_pitch(f0, sr)  # Optional visualization

if __name__ == "__main__":
    main()

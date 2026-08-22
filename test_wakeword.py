import pyaudiowpatch as pyaudio
import numpy as np
from openwakeword.model import Model

# Load the built-in "Hey Jarvis" model
model = Model(
    wakeword_models=[
        "C:/Users/WINDOWS 11/Downloads/openWakeWord-0.6.0/openWakeWord-0.6.0/openwakeword/resources/models/hey_jarvis_v0.1.tflite"
    ]
)

CHUNK = 1280
RATE = 16000

audio = pyaudio.PyAudio()

stream = audio.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK
)

print("🎤 Listening...")
print("Say: Hey Jarvis")

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)

        audio_data = np.frombuffer(data, dtype=np.int16)

        prediction = model.predict(audio_data)

        for name, score in prediction.items():
            if score > 0.5:
                print(f"🔥 Wake word detected! {name} ({score:.2f})")

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()
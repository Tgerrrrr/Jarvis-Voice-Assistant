import os
import re
import time
import tempfile
import wave
from piper import PiperVoice
from datetime import datetime

import pyaudio
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

from openwakeword.model import Model
from faster_whisper import WhisperModel

import requests
import spotipy
from spotipy.oauth2 import SpotifyOAuth

from visualizer import Visualizer


# ============================================================
# SETTINGS
# ============================================================

CHUNK = 1280
RATE = 16000

WAKEWORD_MODEL = (
    "openwakeword/resources/models/hey_jarvis_v0.1.onnx"
)

PIPER_MODEL = "en_US-lessac-medium.onnx"

WHISPER_MODEL = "base"

# Ollama runs locally, no API key needed.
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2"

# How many past exchanges to keep for context (each exchange = 1 user + 1 assistant turn)
MAX_HISTORY_TURNS = 6

SYSTEM_PROMPT = (
    "You are Jarvis, a helpful voice assistant. Keep answers short and "
    "conversational (1-3 sentences) since they will be read aloud. "
    "Address the user as 'sir'. Do not use markdown, bullet points, or "
    "emoji, since this text is converted to speech."
)

SPOTIFY_REDIRECT_URI = "http://127.0.0.1:8888/callback"
SPOTIFY_SCOPES = (
    "user-read-playback-state "
    "user-modify-playback-state "
    "playlist-read-private "
    "playlist-read-collaborative"
)

# Corner for the waveform pop-up: "top-left" | "top-right" | "bottom-left" | "bottom-right"
VISUALIZER_CORNER = "top-left"

# Phrases that end a conversation session and return to wake-word listening.
# Apostrophes are stripped before matching, so "that's all" and "thats all"
# both match.
END_PHRASES = [
    "thats all jarvis",
    "thats all for now",
    "thats all",
    "goodbye",
    "stop listening",
    "im done",
    "thats it jarvis",
]

# How many consecutive silent/empty turns before Jarvis gives up and
# goes back to sleep on its own (in case you walk away mid-conversation).
MAX_CONSECUTIVE_SILENCES = 2


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading Piper voice...")
piper_voice = PiperVoice.load(PIPER_MODEL)
print("Piper voice loaded.")

print("Loading wake-word model...")
wake_model = Model(
    wakeword_models=[WAKEWORD_MODEL],
    inference_framework="onnx"
)
print("Wake-word model loaded.")

print("Loading Whisper...")
whisper = WhisperModel(
    WHISPER_MODEL,
    device="cpu",
    compute_type="int8"
)
print("Whisper loaded.")

print("Checking Ollama...")
ollama_available = False

try:
    check = requests.get("http://localhost:11434", timeout=2)
    ollama_available = True
    print("Ollama is running.")
except requests.exceptions.RequestException:
    print(
        "WARNING: Could not reach Ollama at localhost:11434. "
        "General questions will fail until Ollama is running "
        "(open the Ollama app or run 'ollama serve')."
    )

conversation_history = []  # list of {"role": ..., "content": ...}

print("Connecting to Spotify...")
spotify_client = None
spotify_client_id = os.environ.get("SPOTIFY_CLIENT_ID")
spotify_client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET")

if spotify_client_id and spotify_client_secret:

    try:

        spotify_auth = SpotifyOAuth(
            client_id=spotify_client_id,
            client_secret=spotify_client_secret,
            redirect_uri=SPOTIFY_REDIRECT_URI,
            scope=SPOTIFY_SCOPES,
            cache_path=".spotify_cache"
        )

        spotify_client = spotipy.Spotify(auth_manager=spotify_auth)

        spotify_client.current_user()

        print("Spotify ready.")

    except Exception as e:

        print(f"Spotify auth failed: {e}")
        spotify_client = None

else:

    print(
        "WARNING: SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET not set. "
        "Spotify commands will not work until these are set."
    )

print("Starting waveform pop-up...")
viz = Visualizer(corner=VISUALIZER_CORNER)
viz.start()
print("Waveform pop-up ready.")

pa = pyaudio.PyAudio()


# ============================================================
# TEXT TO SPEECH
# ============================================================

def _play_wav_with_levels(path):

    wf = wave.open(path, "rb")

    stream = pa.open(
        format=pa.get_format_from_width(wf.getsampwidth()),
        channels=wf.getnchannels(),
        rate=wf.getframerate(),
        output=True
    )

    chunk_size = 1024
    data = wf.readframes(chunk_size)

    while data:

        stream.write(data)

        samples = np.frombuffer(data, dtype=np.int16)

        if len(samples) > 0:
            rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2)) / 32768
            viz.push_level(rms * 3)

        data = wf.readframes(chunk_size)

    stream.stop_stream()
    stream.close()
    wf.close()


def speak(text):

    viz.set_state("speaking")

    print(f"Jarvis: {text}")

    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    ) as temp:

        output_file = temp.name

    try:

        with wave.open(output_file, "wb") as wav_file:

            piper_voice.synthesize_wav(
                text,
                wav_file
            )

        _play_wav_with_levels(output_file)

    finally:

        if os.path.exists(output_file):
            os.remove(output_file)

        viz.set_state("idle")


# ============================================================
# RECORD QUESTION
# ============================================================

def listen_for_question():

    viz.set_state("listening")

    print()
    print("🎤 Listening...")

    duration = 5
    frames = []

    def callback(indata, frame_count, time_info, status):

        frames.append(indata.copy())

        rms = np.sqrt(np.mean(indata.astype(np.float64) ** 2)) / 32768
        viz.push_level(rms * 4)

    try:

        with sd.InputStream(
            samplerate=RATE,
            channels=1,
            dtype="int16",
            callback=callback
        ):
            sd.sleep(int(duration * 1000))

        audio_data = np.concatenate(frames, axis=0)

        filename = "question.wav"

        write(
            filename,
            RATE,
            audio_data
        )

        print("Processing...")

        segments, info = whisper.transcribe(
            filename,
            beam_size=5
        )

        text = ""

        for segment in segments:
            text += segment.text

        text = text.strip()

        os.remove(filename)

        print(f"You: {text}")

        return text.lower()

    finally:

        viz.set_state("idle")


# ============================================================
# OLLAMA FALLBACK
# ============================================================

def ask_ollama(question):

    global conversation_history

    if not ollama_available:
        return "I can't reach my local model right now, sir. Ollama doesn't seem to be running."

    conversation_history.append(
        {"role": "user", "content": question}
    )

    max_messages = MAX_HISTORY_TURNS * 2
    if len(conversation_history) > max_messages:
        conversation_history = conversation_history[-max_messages:]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history

    try:

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False
            },
            timeout=60
        )

        response.raise_for_status()

        data = response.json()

        answer = data.get("message", {}).get("content", "").strip()

        if not answer:
            answer = "I'm not sure how to answer that, sir."

        conversation_history.append(
            {"role": "assistant", "content": answer}
        )

        return answer

    except requests.exceptions.RequestException as e:

        print(f"Ollama error: {e}")

        if conversation_history and conversation_history[-1]["role"] == "user":
            conversation_history.pop()

        return "I ran into a problem reaching my local model, sir."


# ============================================================
# SPOTIFY
# ============================================================

def find_playlist_by_name(query):

    results = spotify_client.current_user_playlists(limit=50)
    playlists = results["items"]

    while results["next"]:
        results = spotify_client.next(results)
        playlists.extend(results["items"])

    query_words = set(re.findall(r"\w+", query.lower()))

    best_match = None
    best_score = 0

    for playlist in playlists:

        name = playlist["name"]
        name_words = set(re.findall(r"\w+", name.lower()))

        score = len(query_words & name_words)

        if score > best_score:
            best_score = score
            best_match = playlist

    if best_score == 0:
        return None

    return best_match


def get_active_device_id():

    devices = spotify_client.devices()

    for device in devices["devices"]:
        if device["is_active"]:
            return device["id"]

    if devices["devices"]:
        return devices["devices"][0]["id"]

    return None


def wait_for_device(timeout=15, interval=1.0):

    waited = 0.0

    while waited < timeout:

        device_id = get_active_device_id()

        if device_id:
            return device_id

        time.sleep(interval)
        waited += interval

    return None


def start_playback_with_retry(playlist_uri, attempts=3):

    last_error = None

    for attempt in range(attempts):

        device_id = get_active_device_id()

        if not device_id:
            time.sleep(1.5)
            continue

        try:

            spotify_client.start_playback(
                device_id=device_id,
                context_uri=playlist_uri
            )

            return True

        except Exception as e:

            last_error = e
            print(f"Spotify playback attempt {attempt + 1} failed: {e}")
            time.sleep(1.5)

    if last_error:
        print(f"Spotify playback error after retries: {last_error}")

    return False


def play_playlist(query):

    if not spotify_client:
        return "Spotify isn't set up yet, sir. I don't have API credentials."

    os.startfile("spotify:")

    playlist = find_playlist_by_name(query)

    if not playlist:
        return f"I couldn't find a playlist matching '{query}', sir."

    device_id = wait_for_device()

    if not device_id:
        return "I found the playlist, but there's no active Spotify device, sir."

    success = start_playback_with_retry(playlist["uri"])

    if success:
        return f"Playing {playlist['name']}, sir."

    return "I found the playlist but couldn't start playback, sir."


def extract_playlist_query(command):

    text = command

    text = re.sub(r"\bplay\b", "", text)
    text = re.sub(r"\bmy\b", "", text)
    text = re.sub(r"\bplaylist\b", "", text)
    text = re.sub(r"\bon spotify\b", "", text)
    text = re.sub(r"\bspotify\b", "", text)

    return text.strip()


# ============================================================
# CONVERSATION CONTROL
# ============================================================

def is_end_phrase(command):

    normalized = command.replace("'", "")

    return any(marker in normalized for marker in END_PHRASES)


# ============================================================
# PROCESS COMMAND
# ============================================================

def process_command(command):
    """
    Handles one turn of the conversation.
    Returns True to keep the conversation going, False to end it and
    go back to waiting for the wake word.
    """

    if is_end_phrase(command):

        speak(
            "Alright, sir. Just say Hey Jarvis whenever you need me again."
        )

        return False

    if "what time" in command or "time is it" in command:

        current_time = datetime.now().strftime("%I:%M %p")

        speak(
            f"It is {current_time}, sir."
        )

        return True

    if "clear history" in command or "forget everything" in command:

        conversation_history.clear()

        speak(
            "Conversation history cleared, sir."
        )

        return True

    if "play" in command and ("playlist" in command or "spotify" in command):

        query = extract_playlist_query(command)

        speak(
            f"Looking for your {query} playlist, sir." if query else "Looking for that playlist, sir."
        )

        result = play_playlist(query)

        speak(result)

        return True

    answer = ask_ollama(command)

    speak(answer)

    return True


# ============================================================
# MICROPHONE
# ============================================================

stream = pa.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK
)


# ============================================================
# MAIN LOOP
# ============================================================

print()
print("====================================")
print("          JARVIS IS READY")
print("====================================")
print()
print('Say "Hey Jarvis"')
print()


try:

    while True:

        data = stream.read(
            CHUNK,
            exception_on_overflow=False
        )

        audio_data = np.frombuffer(
            data,
            dtype=np.int16
        )

        prediction = wake_model.predict(
            audio_data
        )

        for name, score in prediction.items():

            if score > 0.5:

                print()
                print(
                    f"🔥 Wake word detected: "
                    f"{name} ({score:.2f})"
                )

                speak(
                    "Yes, how can I help you, sir?"
                )

                # Conversation loop: keep going until an end phrase,
                # or a couple of silent turns in a row.
                consecutive_silences = 0

                while True:

                    command = listen_for_question()

                    if not command:

                        consecutive_silences += 1

                        if consecutive_silences >= MAX_CONSECUTIVE_SILENCES:

                            speak(
                                "I didn't catch anything, sir. Say Hey Jarvis when you're ready."
                            )

                            break

                        continue

                    consecutive_silences = 0

                    keep_going = process_command(command)

                    if not keep_going:
                        break

                wake_model.reset()

                print()
                print('Say "Hey Jarvis" again.')

                break


except KeyboardInterrupt:

    print("\nStopping Jarvis...")


finally:

    stream.stop_stream()
    stream.close()
    pa.terminate()
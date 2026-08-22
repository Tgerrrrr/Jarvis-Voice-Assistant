import os
import re
import time
import tempfile
import wave
import winsound
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

        # Trigger auth flow now (opens browser once, then caches token)
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


# ============================================================
# TEXT TO SPEECH
# ============================================================

def speak(text):

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

        winsound.PlaySound(
            output_file,
            winsound.SND_FILENAME
        )

    finally:

        if os.path.exists(output_file):
            os.remove(output_file)


# ============================================================
# RECORD QUESTION
# ============================================================

def listen_for_question():

    print()
    print("🎤 Listening for your question...")

    duration = 5

    audio = sd.rec(
        int(duration * RATE),
        samplerate=RATE,
        channels=1,
        dtype="int16"
    )

    sd.wait()

    filename = "question.wav"

    write(
        filename,
        RATE,
        audio
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

    # Trim history so the request doesn't grow forever
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

        # Drop the failed user turn so it doesn't poison history
        if conversation_history and conversation_history[-1]["role"] == "user":
            conversation_history.pop()

        return "I ran into a problem reaching my local model, sir."


# ============================================================
# SPOTIFY
# ============================================================

def find_playlist_by_name(query):
    """
    Search the user's own playlists for the best name match against `query`.
    Simple approach: fetch all playlists, score by how many query words
    appear in the playlist name, return the best match.
    """

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

    # No active device — fall back to the first available one
    if devices["devices"]:
        return devices["devices"][0]["id"]

    return None


def wait_for_device(timeout=15, interval=1.0):
    """
    Poll Spotify for an available device until one shows up or timeout
    is reached. Right after launching the app, it can take a few seconds
    to register as a playback device.
    """

    waited = 0.0

    while waited < timeout:

        device_id = get_active_device_id()

        if device_id:
            return device_id

        time.sleep(interval)
        waited += interval

    return None


def start_playback_with_retry(playlist_uri, attempts=3):
    """
    Device IDs can go stale between fetching the device list and issuing
    play, especially right after the app opens. Re-fetch the device and
    retry a couple of times on 404 before giving up.
    """

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

    # Make sure the desktop app is open so there's a device to play on
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
    """
    Pull the playlist name/keywords out of a command like
    'play my british playlist' -> 'british'
    """

    text = command

    text = re.sub(r"\bplay\b", "", text)
    text = re.sub(r"\bmy\b", "", text)
    text = re.sub(r"\bplaylist\b", "", text)
    text = re.sub(r"\bon spotify\b", "", text)
    text = re.sub(r"\bspotify\b", "", text)

    return text.strip()


# ============================================================
# PROCESS COMMAND
# ============================================================

def process_command(command):

    if "what time" in command or "time is it" in command:

        current_time = datetime.now().strftime("%I:%M %p")

        speak(
            f"It is {current_time}, sir."
        )

        return

    if "clear history" in command or "forget everything" in command:

        conversation_history.clear()

        speak(
            "Conversation history cleared, sir."
        )

        return

    if "goodbye" in command or "stop listening" in command:

        speak(
            "Goodbye, sir."
        )

        return

    if "play" in command and ("playlist" in command or "spotify" in command):

        query = extract_playlist_query(command)

        speak(
            f"Looking for your {query} playlist, sir." if query else "Looking for that playlist, sir."
        )

        result = play_playlist(query)

        speak(result)

        return

    # Anything else goes to the local model
    answer = ask_ollama(command)

    speak(answer)


# ============================================================
# MICROPHONE
# ============================================================

audio = pyaudio.PyAudio()

stream = audio.open(
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

                # Listen for the question
                command = listen_for_question()

                if command:
                    process_command(command)

                wake_model.reset()

                print()
                print('Say "Hey Jarvis" again.')

                break


except KeyboardInterrupt:

    print("\nStopping Jarvis...")


finally:

    stream.stop_stream()
    stream.close()
    audio.terminate()
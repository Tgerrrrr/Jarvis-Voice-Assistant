# Jarvis Voice Assistant — Documentation

## Overview

Jarvis is a local, wake-word-activated voice assistant running on Windows. It listens continuously for the phrase "Hey Jarvis," records a spoken question, transcribes it, generates a response, and speaks the answer back. All processing (speech recognition, language model, text-to-speech) runs locally on the machine, with the exception of Spotify playback control, which uses Spotify's cloud API.

## Architecture

The pipeline has five stages:

1. **Wake word detection** — openWakeWord, using the `hey_jarvis_v0.1` ONNX model. Runs continuously on live microphone input in 1280-sample chunks.
2. **Speech-to-text** — faster-whisper, `base` model, running on CPU with int8 quantization. Records a fixed 5-second window after wake word detection and transcribes it.
3. **Language processing** — Ollama, running the `llama3.2` model locally. Handles any question that doesn't match a hardcoded command. Maintains a rolling conversation history (last 6 exchanges).
4. **Text-to-speech** — Piper, using the `en_US-lessac-medium` voice. Synthesizes responses to a temporary WAV file and plays it via Windows' built-in `winsound`.
5. **Music control** — Spotify Web API via the `spotipy` library. Handles playlist search and playback commands separately from the language model.

## Request flow

1. Microphone stream runs continuously; each audio chunk is scored against the wake word model.
2. On detection (score > 0.5), Jarvis speaks a greeting and records a 5-second question.
3. The question is transcribed and lowercased.
4. `process_command()` checks the text against hardcoded intents in order:
   - Time query
   - Clear conversation history
   - Goodbye
   - Spotify playlist playback (contains "play" plus "playlist" or "spotify")
5. If nothing matches, the question is sent to Ollama with conversation history and a system prompt tuned for short, speech-friendly answers.
6. The response is spoken aloud.

## Hardcoded commands

| Command trigger | Behavior |
|---|---|
| "what time" / "time is it" | Speaks current time, no LLM call |
| "clear history" / "forget everything" | Clears conversation history |
| "goodbye" / "stop listening" | Speaks farewell |
| "play [x] playlist" (+ "spotify") | Searches user's playlists, starts playback |

Anything else falls through to the local LLM.

## Spotify integration

- Registered as a Spotify Developer app (Development mode), redirect URI `http://127.0.0.1:8888/callback`.
- Scopes used: `user-read-playback-state`, `user-modify-playback-state`, `playlist-read-private`, `playlist-read-collaborative`.
- Requires Spotify Premium — the Web API can only control remote playback on Premium accounts.
- Playlist matching: fetches all user playlists (with pagination), scores each by word overlap between the spoken query and playlist name, picks the highest-scoring match. No fuzzy/partial matching beyond whole-word overlap.
- Playback startup: launches the Spotify desktop app via `spotify:` URI, polls for an available device (up to 15 seconds) rather than using a fixed delay, and retries playback up to 3 times if the device ID goes stale between calls.

## Environment variables required

| Variable | Purpose |
|---|---|
| `SPOTIFY_CLIENT_ID` | Spotify Developer app client ID |
| `SPOTIFY_CLIENT_SECRET` | Spotify Developer app client secret |

None of the other components (wake word, Whisper, Ollama, Piper) require API keys — all run locally.

## Runtime requirements

- Ollama installed and running (`ollama serve` or the desktop app), with the `llama3.2` model pulled.
- Spotify desktop app installed, logged into a Premium account.
- Python environment with: `pyaudio`, `numpy`, `sounddevice`, `scipy`, `openwakeword`, `faster-whisper`, `piper-tts`, `requests`, `spotipy`.
- Windows OS (uses `winsound` for audio playback).

## Known limitations

- Fixed 5-second recording window for questions; no silence detection, so longer questions get cut off and short pauses waste recording time.
- `llama3.2` (3B parameters) has weaker reasoning than hosted models, no knowledge of anything past its training data, and no live internet access — cannot answer questions about current events, weather, or anything requiring up-to-date information.
- Playlist name matching is word-overlap based, not exact. Similarly named playlists can be mismatched.
- Requires Ollama running locally at all times — if the service isn't active, general questions fail with a spoken error instead of a crash.
- Entire assistant only runs while the laptop is on; there is no standalone hardware yet.

## Out of scope for this document

Planned migration to a microcontroller-based edge device (ESP32-S3 for wake word detection, microphone, speaker, and later camera and LED status indicators, with the laptop remaining the server for Whisper/Ollama/Piper/Spotify) is a separate, not-yet-implemented phase and is not covered here.

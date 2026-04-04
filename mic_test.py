import argparse
import numpy as np
import pyaudio
from openwakeword.model import Model

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_path",
    type=str,
    required=True,
    help="Full path to custom .tflite model"
)
parser.add_argument(
    "--threshold",
    type=float,
    default=0.5,
    help="Detection threshold"
)
parser.add_argument(
    "--chunk_size",
    type=int,
    default=1280,
    help="Audio chunk size in samples (1280 = 80 ms at 16 kHz)"
)
args = parser.parse_args()

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = args.chunk_size

audio = pyaudio.PyAudio()

mic_stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK
)

model = Model(
    wakeword_models=[args.model_path],
    inference_framework="tflite"
)

model_name = list(model.models.keys())[0]

print("\nListening for wakeword...")
print(f"Model: {model_name}")
print(f"Threshold: {args.threshold}\n")

try:
    while True:
        audio_data = np.frombuffer(mic_stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
        prediction = model.predict(audio_data)

        score = prediction[model_name]
        detected = score >= args.threshold

        print(f"\rScore: {score:.6f} | {'DETECTED' if detected else '---'}", end="")

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    mic_stream.stop_stream()
    mic_stream.close()
    audio.terminate()

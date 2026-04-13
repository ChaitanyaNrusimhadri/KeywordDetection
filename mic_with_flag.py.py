import argparse
import numpy as np
import pyaudio
from openwakeword.model import Model


def listen_for_wakeword(model_path, threshold=0.5, chunk_size=1280):
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = chunk_size

    audio = pyaudio.PyAudio()

    mic_stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    model = Model(
        wakeword_models=[model_path],
        inference_framework="tflite"
    )

    model_name = list(model.models.keys())[0]

    print("\nListening for wakeword...")
    print(f"Model: {model_name}")
    print(f"Threshold: {threshold}\n")

    try:
        while True:
            audio_data = np.frombuffer(
                mic_stream.read(CHUNK, exception_on_overflow=False),
                dtype=np.int16
            )

            prediction = model.predict(audio_data)
            score = prediction[model_name]
            detected = score >= threshold

            print(f"\rScore: {score:.6f} | {'DETECTED' if detected else '---'}", end="")

            if detected:
                print("\nWakeword detected. Exiting...")
                return True

    except KeyboardInterrupt:
        print("\nStopped by user.")
        return False

    finally:
        mic_stream.stop_stream()
        mic_stream.close()
        audio.terminate()


if __name__ == "__main__":
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

    result = listen_for_wakeword(
        model_path=args.model_path,
        threshold=args.threshold,
        chunk_size=args.chunk_size
    )

    print(f"Result: {result}")
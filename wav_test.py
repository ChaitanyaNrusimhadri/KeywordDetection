from openwakeword.model import Model

MODEL_PATH = "/home/chaitu/KeywordDetection/models/dayn_jer_di_tected.tflite"
AUDIO_PATH = "/home/chaitu/KeywordDetection/wav_audios/DangerDetected.wav"

model = Model(
	wakeword_models=[MODEL_PATH],
	inference_framework="tflite"
)

preds = model.predict_clip(AUDIO_PATH)

max_score = max(x["dayn_jer_di_tected"] for x in preds)
detected = any(x["dayn_jer_di_tected"] >= 0.5 for x in preds)

print("Max Score: ", max_score)
print("Detected: ", detected)
#print(preds)


	
	
	
		
		

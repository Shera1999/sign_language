# Cell 1: Imports & Model Initialization

import cv2
import numpy as np
import onnxruntime as ort
import pyttsx3
import time

# Choose the appropriate model file:
# - 'signcnn_qdq.onnx'  for local macOS testing (QLinearConv ops)
# - 'signcnn_quant.onnx' for Raspberry Pi deployment (ConvInteger ops)
model_path = 'signcnn_qdq.onnx'

# Create an ONNX Runtime session
sess = ort.InferenceSession(model_path)
inp_name = sess.get_inputs()[0].name

# Prepare label mapping: A–Y excluding J,Z
labels = [chr(c) for c in range(65, 91) if c not in (74, 90)]

# Text-to-Speech engine
tts = pyttsx3.init()
tts.setProperty('rate', 150)

print(f"Loaded ONNX model from `{model_path}`")
# Cell 2: Live Video Inference Loop

cap = cv2.VideoCapture(0)
prev_letter = None
conf_thresh = 0.8

while True:
    t0 = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # 1) Preprocess frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    m = min(h, w)
    crop = gray[(h-m)//2:(h+m)//2, (w-m)//2:(w+m)//2]
    small = cv2.resize(crop, (28,28)).astype('float32') / 255.0
    inp = small.reshape(1, 1, 28, 28)

    # 2) ONNX inference
    out = sess.run(None, {inp_name: inp})[0][0]
    idx = np.argmax(out)
    conf = out[idx] / out.sum()
    letter = labels[idx]

    # 3) Overlay prediction + confidence
    text = f"{letter} ({conf*100:.1f}%)"
    cv2.putText(frame, text, (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # 4) Speak new high-confidence letter
    if conf > conf_thresh and letter != prev_letter:
        tts.say(letter)
        tts.runAndWait()
        prev_letter = letter

    # 5) Compute & display FPS
    fps = 1.0 / (time.time() - t0)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10,60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

    # 6) Show and quit on ESC
    cv2.imshow('Live Sign Recognition', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

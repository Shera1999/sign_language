#!/usr/bin/env python3
import cv2
import numpy as np
import onnxruntime as ort
import pyttsx3
import time
import sys

def open_camera():
    # Try AVFoundation and QuickTime backends on macOS, indices 0–3
    backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_QT]
    for b in backends:
        for idx in range(4):
            cap = cv2.VideoCapture(idx, b)
            if cap.isOpened():
                print(f"✅ Opened camera index={idx}, backend={b}")
                return cap
            cap.release()
    return None

def main():
    # 1) Model and TTS init
    model_path = 'signcnn_qdq.onnx'  # use signcnn_quant.onnx on Pi
    sess       = ort.InferenceSession(model_path)
    inp_name   = sess.get_inputs()[0].name
    labels     = [chr(c) for c in range(65, 91) if c not in (74, 90)]
    tts        = pyttsx3.init()
    tts.setProperty('rate', 150)

    # 2) Open camera
    cap = open_camera()
    if cap is None:
        print("❌ ERROR: Unable to open any camera.")
        print(" • Make sure no other app is using it.")
        print(" • Grant Terminal/IDE camera permission in System Settings → Privacy & Security → Camera")
        sys.exit(1)

    # Optionally, set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prev_letter = None
    conf_thresh = 0.8

    # 3) Main loop
    while True:
        start, (ret, frame) = time.time(), cap.read()
        if not ret:
            print("⚠️ Frame read failed, retrying…")
            time.sleep(0.1)
            continue

        # Preprocess
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        m = min(h, w)
        crop = gray[(h-m)//2:(h+m)//2, (w-m)//2:(w+m)//2]
        small = cv2.resize(crop, (28, 28)).astype('float32') / 255.0
        inp = small.reshape(1, 1, 28, 28)

        # Inference
        out = sess.run(None, {inp_name: inp})[0][0]
        idx  = int(np.argmax(out))
        conf = float(out[idx] / out.sum())
        letter = labels[idx]

        # Overlay
        cv2.putText(frame, f"{letter} ({conf*100:.1f}%)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Speak if new high-confidence
        if conf > conf_thresh and letter != prev_letter:
            tts.say(letter)
            tts.runAndWait()
            prev_letter = letter

        # FPS
        fps = 1.0 / (time.time() - start)
        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Display & exit on ESC
        cv2.imshow('Live Sign Recognition', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

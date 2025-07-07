# Real-Time Sign Language Recognition Pipeline

This repository demonstrates a workflow for training, exporting, quantizing, and deploying a Convolutional Neural Network (CNN) that recognizes static American Sign Language letters (A–Y, excluding J & Z) in real time. The dataset used is [Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist). It’s structured into three main components:

1. `cnn.ipynb` — Dataset exploration & CNN training  
2. `02-export_quantize_deploy.ipynb` — ONNX export & quantization (dynamic & static QDQ)  
3. `03-live_inference_edge.py` — Live webcam inference script (macOS & Raspberry Pi)

---

## Repository Structure


.
├── cnn.ipynb
├── 02-export\_quantize\_deploy.ipynb
├── 03-live\_inference\_edge.py
├── requirements.txt
├── sign\_mnist\_train.csv
├── sign\_mnist\_test.csv
├── best\_signcnn.pth
└── README.md


### Files

- **cnn.ipynb**  
  - Loads and explores the Sign Language MNIST CSV files  
  - Defines a PyTorch `SignCNN` model  
  - Trains on 27 k samples (train set) and evaluates on 7 k (test set)  
  - Saves best weights to `best_signcnn.pth`

- **02-export_quantize_deploy.ipynb**  
  - Imports the trained PyTorch model  
  - Exports to ONNX (`signcnn.onnx`, FP32)  
  - Applies dynamic quantization and static QDQ quantization, producing:  
    - `signcnn_quant.onnx` (dynamic; ConvInteger ops for edge devices)  
    - `signcnn_qdq.onnx` (static QDQ; QLinearConv ops for macOS)  
  - Verifies accuracy on a test batch and benchmarks inference latency

- **03-live_inference_edge.py**  
  - Loads an ONNX model and runs a live webcam loop at ~20 FPS  
  - Preprocesses frames to 28×28 grayscale for ONNX Runtime inference  
  - Overlays predicted letter + confidence, with optional Text-to-Speech  
  - Auto-selects the correct model:  
    - macOS: `signcnn_qdq.onnx` (QLinearConv supported by pip wheel)  
    - Raspberry Pi: `signcnn_quant.onnx` (ConvInteger supported by ARM wheel)

---

## Usage

### 1. Train your CNN

Open `cnn.ipynb` and run all cells to:

   * Visualize data & baseline models
   * Define & train `SignCNN` in PyTorch
   * Save checkpoints to `best_signcnn.pth`

### 2. Export & Quantize

1. In the same Jupyter session, open `02-export_quantize_deploy.ipynb` to:

   * Load `best_signcnn.pth`
   * Export to ONNX (FP32) → `signcnn.onnx`
   * Quantize models:

     * Dynamic → `signcnn_quant.onnx`
     * Static QDQ → `signcnn_qdq.onnx`
   * Run accuracy checks & benchmark inference

### 3. Live Inference

* **Local dev (macOS)**

  ```bash
  python 03-live_inference_edge.py
  ```

  Uses `signcnn_qdq.onnx`.

* **Edge deploy (Raspberry Pi)**

  1. Copy `signcnn_quant.onnx` & `03-live_inference_edge.py` to the Pi (e.g., via `scp`).
  2. On the Pi:

     ```bash
     sudo apt update
     sudo apt install python3-pip python3-opencv
     pip3 install onnxruntime numpy pyttsx3
     python3 03-live_inference_edge.py
     ```

     Automatically switches to `signcnn_quant.onnx` for ConvInteger ops.

---

## Results & Next Steps

* Per-letter accuracy ≥ 99 % on static test images
* Live webcam performance: \~20 FPS on macOS (QDQ model)

### Extensions

* Expand to full ASL word/phrase recognition (video sequences)
* Integrate hand-detection (MediaPipe/SSD) for auto-cropping
* Deploy via Flask or React Native for remote use

```
```

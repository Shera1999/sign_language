```markdown
# Real-Time Sign Language Recognition Pipeline

This repository demonstrates a workflow for training, exporting, quantizing, and deploying a Convolutional Neural Network (CNN) that recognizes static American Sign Language letters (A–Y, excluding J & Z) in real time. The dataset used in this project is from: https://www.kaggle.com/datasets/datamunge/sign-language-mnist. It’s structured into three main components:

1. **`cnn.ipynb`** — Dataset exploration & CNN training  
2. **`02-export_quantize_deploy.ipynb`** — ONNX export & quantization (QDQ & dynamic)  
3. **`03-live_inference_edge.py`** — Live webcam inference script (macOS & Raspberry Pi)

---

## Repository Structure

```

.
├── cnn.ipynb
├── 02-export\_quantize\_deploy.ipynb
├── 03-live\_inference\_edge.py
├── requirements.txt
├── sign\_mnist\_train.csv
├── sign\_mnist\_test.csv
├── best\_signcnn.pth
└── README.md

```

- **`cnn.ipynb`**  
  - Loads and explores the Sign-Language MNIST CSVs  
  - Defines a PyTorch `SignCNN` model  
  - Trains on the 27 k-sample train split, evaluates on 7 k test split  
  - Saves best weights to `best_signcnn.pth`

- **`02-export_quantize_deploy.ipynb`**  
  - Imports the trained PyTorch model  
  - Exports to **ONNX** (`signcnn.onnx`, FP32)  
  - Applies **dynamic** and **static QDQ** quantization to produce:
    - `signcnn_quant.onnx` (dynamic → ConvInteger ops for edge)  
    - `signcnn_qdq.onnx`  (static QDQ → QLinearConv for local macOS)  
  - Verifies accuracy on a small test batch and benchmarks inference latency  

- **`03-live_inference_edge.py`**  
  - Loads an ONNX model and runs a live webcam loop at ~20 FPS  
  - Preprocesses each frame to 28×28 grayscale, runs ONNX Runtime inference  
  - Overlays predicted letter + confidence and optional Text-to-Speech  
  - Automatically picks the right model for:
    - **macOS**: `signcnn_qdq.onnx` (QLinearConv supported by pip wheel)  
    - **Raspberry Pi**: `signcnn_quant.onnx` (ConvInteger supported by ARM wheel)

---

## Usage

### 1. Train your CNN (`cnn.ipynb`)

* Launch Jupyter:

  ```bash
  jupyter notebook
  ```
* Open `cnn.ipynb`, run all cells:

  * Data visualization & baseline models
  * Define & train `SignCNN` in PyTorch
  * Model checkpoints saved to `best_signcnn.pth`

### 2. Export & Quantize (`02-export_quantize_deploy.ipynb`)

* In the same Jupyter session:

  * Load `best_signcnn.pth`
  * Export to ONNX (FP32) → `signcnn.onnx`
  * Quantize (dynamic → `signcnn_quant.onnx`; static QDQ → `signcnn_qdq.onnx`)
  * Run accuracy checks and benchmark inference times

### 3. Live Inference (`03-live_inference_edge.py`)

* **Local dev (macOS)**

  ```bash
  python 03-live_inference_edge.py
  ```

  → Uses `signcnn_qdq.onnx` for live webcam inference.

* **Edge deploy (Raspberry Pi)**

  1. Copy `signcnn_quant.onnx` & `03-live_inference_edge.py` to the Pi (via `scp`).
  2. On the Pi:

     ```bash
     sudo apt update
     sudo apt install python3-pip python3-opencv
     pip3 install onnxruntime numpy pyttsx3
     python3 03-live_inference_edge.py
     ```

     → Switches to `signcnn_quant.onnx` automatically for ConvInteger ops.

---

## Results & Next Steps

* **Per-letter accuracy** ≥ 99% on static test images
* **Live webcam** performance: \~20 FPS on macOS (QDQ model)
* **Extensions**

  * Expand to full ASL word/phrase recognition (video sequences)
  * Integrate hand‐detection (MediaPipe/SSD) for auto‐cropping
  * Deploy via Flask or React Native for remote use


```
```

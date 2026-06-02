#Vilence Detection
# 🎥 Lightweight Temporal Violence Detection Using YOLOv7-Tiny and Bi-GRU

<p align="center">
  <img src="https://img.shields.io/badge/Accuracy-83.02%25-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Model_Size-65.3_MB-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Inference-19.75ms%2Fseq-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Published-ScienceDirect-red?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-CC_BY--NC--ND_4.0-lightgrey?style=for-the-badge" />
</p>

<p align="center">
  <a href="https://www.sciencedirect.com/science/article/pii/S2772941926000578">
    📄 Read the Paper (ScienceDirect)
  </a>
  &nbsp;|&nbsp;
  <a href="https://doi.org/10.1016/j.sasc.2026.200494">
    🔗 DOI: 10.1016/j.sasc.2026.200494
  </a>
</p>

> **Official implementation** of the paper:  
> **"Lightweight Temporal Violence Detection Using YOLOv7-Tiny and Bi-GRU for Resource-Constrained Environments"**  
> *Published in **Systems and Soft Computing**, Vol. 8 (2026), Article 200494 — Elsevier*

---

## 👥 Authors

**Hasanur Rahman Shishir¹, Nazmun Nahar¹, Mayeen Uddin Khandaker²³⁴\*, Md Hasan Imam¹\*, Md. Iftekharul Alam Efat¹, Mohammad Kamrul Hasan⁵**

¹ Institute of Information Technology, Noakhali Science and Technology University (NSTU), Noakhali, Bangladesh  
² Applied Physics and Radiation Technologies Group, CCDCU, Faculty of Engineering and Technology, Sunway University, Selangor, Malaysia  
³ Miyan Research Institute, International University of Business Agriculture and Technology (IUBAT), Dhaka, Bangladesh  
⁴ Department of Physics, College of Science, Korea University, Seoul, Republic of Korea  
⁵ Information and Communication Engineering, NSTU, Noakhali, Bangladesh  

\* Corresponding authors: `mayeenk@iubat.edu` | `hasan.imam@nstu.edu.bd`

---

## 📌 Overview

The rapid proliferation of surveillance cameras has generated volumes of video data that are impossible to monitor manually. Existing deep learning violence detection systems are accurate but rely on computationally heavy architectures that cannot run in resource-constrained edge environments.

This paper proposes a **lightweight spatiotemporal violence detection framework** with three key novelties:

- ⚡ **YOLOv7-Tiny** — a condensed 5-block convolutional backbone (Conv + BatchNorm + SiLU) for efficient per-frame spatial feature extraction (512-dim output vector)
- 🔁 **Bi-GRU** — a Bidirectional Gated Recurrent Unit (hidden size: 128, layers: 2, dropout: 0.6) for temporal modeling in both forward and backward directions, with a **frame-level attention mechanism**
- 🎛️ **Adaptive Frame Control (AFC)** — dynamically adjusts input frame rates based on available hardware compute, eliminating latency accumulation in real-time streams
- 🔀 **Multi-Source Prefetcher (MSP)** — handles heterogeneous inputs (RTSP streams, webcams, uploaded video files) in a unified queue

Additional training techniques: **mixed-precision training (FP16)** with automatic loss scaling, **cosine annealing** learning rate scheduler with warm restarts, and a **weighted cross-entropy loss** to handle class imbalance.

---

## 🏆 Key Results

Evaluated on a balanced subset of the **UCF-Crime dataset** (280 videos, binary classification):

| Metric                    | Value          |
|---------------------------|----------------|
| **Accuracy**              | **83.02%**     |
| **Precision**             | **83.34%**     |
| **Recall**                | **83.02%**     |
| **F1-Score**              | **82.91%**     |
| **AUC**                   | **80.14%**     |
| Inference Time (per seq.) | 19.75 ms       |
| Inference Time (per frame)| 1.23 ms        |
| Throughput                | 50.6 seq/sec   |
| Peak GPU Memory           | ~404 MB        |
| Model Size                | 65.3 MB        |
| Parameters                | 16.33 M        |

> 💡 Mixed-precision training (FP16) reduced total training time by **38%**.

---

## 🏗️ Model Architecture

```
Input Video (RTSP / Webcam / File)
           │
  Multi-Source Prefetcher (MSP)
   [Unified frame queue, sync'd FPS]
           │
  Adaptive Frame Control (AFC)
   [Compares Ta vs Ds; drops stale
    frames to prevent latency buildup]
           │
  Preprocessing per Frame (224×224)
   ├── Resize → 224×224
   ├── Pixel Normalization [0,1]
   ├── Gaussian Blur (noise reduction)
   └── CLAHE (local contrast enhancement)
           │
  ┌────────▼────────────────────────────┐
  │       YOLOv7-Tiny Feature Extractor │  ← Spatial Stage
  │                                     │
  │  Input (224×224×3)                  │
  │  → Conv Block 1 (112×112×32)        │
  │  → Conv Block 2 (56×56×64)          │
  │  → Conv Block 3 (28×28×128)         │
  │  → Conv Block 4 (14×14×256)         │
  │  → Conv Block 5 (7×7×512)           │
  │  → Global Avg Pool + Flatten        │
  │  → Feature Vector (512-dim)         │
  └────────┬────────────────────────────┘
           │  Sequence of 8 frame features
  ┌────────▼────────────────────────────┐
  │  Bi-GRU with Frame Attention        │  ← Temporal Stage
  │                                     │
  │  → Forward GRU  ──┐                 │
  │  → Backward GRU ──┴→ Context Vector │
  │  [Hidden: 128, Layers: 2, Drop: 0.6]│
  │  → Frame Attention: α = tanh(Wa·h+ba)│
  └────────┬────────────────────────────┘
           │
  Fully Connected Classifier
           │
  ┌────────┴────────┐
  │    Violence     │    Non-Violence
  └─────────────────┘
           │
  Alert System (Alarm / Notification)
```

---

## 📂 Dataset

The model uses a **processed binary subset of the UCF-Crime dataset**:

| Property              | Value                        |
|-----------------------|------------------------------|
| Source Dataset        | UCF-Crime                    |
| Total Videos          | 240                          |
| Classes               | 2 (Normal, Violence)         |
| Normal Videos         | 124 (47.0%)                  |
| Violence Videos       | 140 (53.0%)                  |
| Spatial Resolution    | 320 × 240 px                 |
| Input to Model        | Resized to 224 × 224         |
| Sequence Length       | 8 frames per video           |
| Frame Stride          | 2                            |
| Total Sequences       | 264                          |
| Total Frames          | 2,112                        |
| Train / Test Split    | 80:20 (stratified)           |

**Violence categories** from UCF-Crime merged into the `Violence` class: Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting, Robbery, Shooting, Shoplifting, Stealing, Road Accident, Vandalism.

> 📦 Original UCF-Crime dataset: [Sultani et al., CVPR 2018](https://www.crcv.ucf.edu/projects/real-world/)  
> 📦 Processed subset used in this study: Available via [Dropbox link in paper]

---

## 🖥️ Hardware Tested

The framework was benchmarked across four hardware environments:

| Platform                          | Description                                              |
|-----------------------------------|----------------------------------------------------------|
| **NVIDIA Tesla T4 GPU** (Colab)   | GPU-accelerated cloud inference (high throughput)        |
| **Intel Xeon CPU ~2.20 GHz** (Colab) | CPU-only cloud inference (GPU disabled)               |
| **Intel Core i5-6300U** (Edge)    | 2.4 GHz dual-core, 8 GB RAM, no dedicated GPU           |
| **Apple MacMini M1**              | 8-core Apple M1, 8 GB unified memory, Neural Engine      |

---

## 📊 Comparison with State-of-the-Art (UCF-Crime)

| Method                       | Accuracy | Precision | Recall  | F1-Score | AUC     |
|------------------------------|----------|-----------|---------|----------|---------|
| 3D CNN                       | 52.21%   | 52.02%    | 50.02%  | 50.98%   | 52.83%  |
| DEARESt (VGG19 + FlowNet)    | 60.00%   | 63.91%    | 60.00%  | 60.61%   | 59.47%  |
| VGG16                        | 72.50%   | 77.71%    | 72.50%  | 72.86%   | 76.67%  |
| Attention Residual LSTM      | 78.43%   | 78.00%    | 78.50%  | 78.25%   | 78.60%  |
| **YOLOv7-Tiny + Bi-GRU (Ours)** | **83.02%** | **83.34%** | **83.02%** | **82.91%** | **80.14%** |

### Broader Comparison (Efficiency vs. Accuracy)

| Model                        | Accuracy | Comp. Cost | Model Size | Parameters |
|------------------------------|----------|------------|------------|------------|
| MobileNetV2 + Attention LSTM | 78.43%   | Low        | 12.8 MB    | 3.3 M      |
| DEARESt (VGG19 + FlowNet)    | 76.66%   | High       | 1187.5 MB  | 305.49 M   |
| Siamese 3D CNN               | 38.4%    | Low        | 33.6 MB    | 2.8 M      |
| 3D CNN (General)             | 90%+     | Very High  | 2647.70 MB | 297.56 M   |
| **YOLOv7-Tiny + Bi-GRU (Ours)** | **83.02%** | **Low** | **65.3 MB** | **16.33 M** |

---

## 💻 Software Stack & Deployment

The system is built with a **modular frontend–backend architecture**:

- **Frontend:** [Streamlit](https://streamlit.io/) — accepts video via file upload or live webcam stream; displays predictions in real-time
- **Backend:** [FastAPI](https://fastapi.tiangolo.com/) — REST API inference engine; loads model once at startup; handles frame preprocessing (resize, normalize, tensor conversion) and returns JSON predictions asynchronously
- **Frame processing:** OpenCV — frame extraction and serialization
- **Communication:** HTTP POST (frames from frontend → backend → JSON predictions)

> ⚠️ Full edge deployment on devices like NVIDIA Jetson platforms is planned as future work but not yet implemented.

---

## 🛠️ Requirements

```bash
Python >= 3.8
torch >= 1.12
torchvision
opencv-python          # Frame extraction & preprocessing
numpy
scikit-learn           # Metrics, train/test split
matplotlib
tqdm
fastapi                # Backend inference server
uvicorn                # ASGI server for FastAPI
streamlit              # Frontend UI
requests               # HTTP communication
Pillow
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## 📁 Repository Structure

```
YOLOv7-Tiny-BiGRU-Violence-Detection/
│
├── data/
│   ├── raw/                            # Original UCF-Crime videos
│   └── processed/
│       ├── normal/                     # Normal class videos
│       └── violence/                   # Violence class videos
│
├── models/
│   ├── yolov7_tiny_extractor.py        # 5-block YOLOv7-Tiny backbone (512-dim output)
│   ├── bigru_attention.py              # Bi-GRU (hidden=128, layers=2, dropout=0.6)
│   │                                   # + frame-level attention mechanism
│   └── pipeline.py                     # Full MSP → AFC → YOLOv7-Tiny → Bi-GRU pipeline
│
├── preprocessing/
│   ├── frame_extractor.py              # Video → 8-frame sequences (stride=2)
│   ├── resize_normalize.py             # Resize to 224×224 + normalize [0,1]
│   ├── gaussian_blur.py                # High-frequency noise reduction
│   └── clahe.py                        # Local contrast enhancement
│
├── afc/
│   ├── multi_source_prefetcher.py      # MSP: RTSP / webcam / file unified queue
│   └── adaptive_frame_control.py       # AFC: Ta vs Ds comparison + frame dropping
│
├── training/
│   ├── train.py                        # Mixed-precision (FP16) training loop
│   ├── loss.py                         # Weighted cross-entropy for class imbalance
│   └── config.yaml                     # Epochs, batch size, LR, cosine annealing
│
├── evaluation/
│   ├── evaluate.py                     # Accuracy, Precision, Recall, F1, AUC
│   ├── confusion_matrix.py             # Confusion matrix plot
│   └── roc_curve.py                    # ROC + AUC plot
│
├── xai/
│   └── gradcam.py                      # Grad-CAM visualization for violence/normal
│
├── deployment/
│   ├── backend/
│   │   └── app.py                      # FastAPI inference server
│   └── frontend/
│       └── app.py                      # Streamlit UI (upload / webcam)
│
├── weights/
│   └── final_yolov7tiny_bigru_model.pth
│
├── results/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── training_curves_fold1.png
│   └── training_curves_fold2.png
│
├── notebooks/
│   ├── training.ipynb
│   └── evaluation.ipynb
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Usage

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/YOLOv7-Tiny-BiGRU-Violence-Detection.git
cd YOLOv7-Tiny-BiGRU-Violence-Detection
```

### 2. Download & Prepare the Dataset

Download the UCF-Crime dataset and set up the binary processed subset:

```bash
# Organize videos into:
data/processed/normal/      ← normal surveillance videos
data/processed/violence/    ← violence class videos (merged anomaly categories)
```

### 3. Preprocess & Extract Frame Sequences

```bash
python preprocessing/frame_extractor.py \
  --input data/processed \
  --output data/sequences \
  --seq_len 8 \
  --stride 2 \
  --resize 224
```

### 4. Train the Model

```bash
python training/train.py \
  --config training/config.yaml \
  --mixed_precision \
  --split 0.8
```

### 5. Evaluate the Model

```bash
python evaluation/evaluate.py \
  --checkpoint weights/final_yolov7tiny_bigru_model.pth \
  --data data/sequences/test
```

### 6. Generate Grad-CAM Visualizations

```bash
python xai/gradcam.py \
  --video path/to/video.mp4 \
  --checkpoint weights/final_yolov7tiny_bigru_model.pth \
  --class violence
```

### 7. Run the Full Deployment System

**Start the FastAPI backend:**
```bash
cd deployment/backend
uvicorn app:app --host 0.0.0.0 --port 8000
```

**Launch the Streamlit frontend:**
```bash
cd deployment/frontend
streamlit run app.py
```

Then open `http://localhost:8501` in your browser to upload videos or stream from a webcam.

---

## 📖 Citation

If you find this work useful, please cite:

```bibtex
@article{shishir2026violence,
  title     = {Lightweight temporal violence detection using YOLOv7-Tiny and Bi-GRU
               for resource-constrained environments},
  author    = {Shishir, Hasanur Rahman and Nahar, Nazmun and Khandaker, Mayeen Uddin
               and Imam, Md Hasan and Efat, Md. Iftekharul Alam and Hasan, Mohammad Kamrul},
  journal   = {Systems and Soft Computing},
  volume    = {8},
  pages     = {200494},
  year      = {2026},
  publisher = {Elsevier},
  doi       = {10.1016/j.sasc.2026.200494},
  url       = {https://www.sciencedirect.com/science/article/pii/S2772941926000578}
}
```

---

## 🔮 Future Work

As outlined in the paper:

- [ ] Test on larger and more diverse real-world surveillance datasets
- [ ] Validate actual inference performance on physical edge devices (e.g., NVIDIA Jetson Nano/Xavier)
- [ ] Extend to multimodal detection incorporating audio signals alongside video
- [ ] Implement multi-class anomaly detection beyond binary violence/normal
- [ ] Further model compression (quantization, pruning) for sub-10 MB deployment
- [ ] Explore advanced lightweight temporal architectures (e.g., Video Swin Transformer variants)

---

## 📜 License

This work is published as open access under the **CC BY-NC-ND 4.0** license.  
© 2026 The Authors. Published by Elsevier B.V.  
See: [http://creativecommons.org/licenses/by-nc-nd/4.0/](http://creativecommons.org/licenses/by-nc-nd/4.0/)

---

## 🙏 Acknowledgements

The authors extend their appreciation to **NSTU (Noakhali Science and Technology University), Bangladesh**.

> *This research received no external funding.*

---

<p align="center">Made with ❤️ for safer public environments</p>

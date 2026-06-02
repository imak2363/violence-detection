# Violence-detection
# 🎥 Lightweight Temporal Violence Detection Using YOLOv7-Tiny and Bi-GRU

<p align="center">
  <img src="https://img.shields.io/badge/Accuracy-XX.XX%25-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Model_Size-Lightweight-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Real--Time-Capable-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Published-ScienceDirect-red?style=for-the-badge" />
</p>

<p align="center">
  <a href="https://www.sciencedirect.com/science/article/pii/S2772941926000578">
    📄 Read the Paper (ScienceDirect)
  </a>
</p>

> **Official implementation** of the paper:  
> **"Lightweight Temporal Violence Detection Using YOLOv7-Tiny and Bi-GRU for Resource-Constrained Environments"**  
> *Published in Systems and Soft Computing — Elsevier (2026)*

---

## 📌 Overview

Automated violence detection in surveillance systems is critical for public safety, yet most state-of-the-art models are too computationally heavy for resource-constrained devices such as edge cameras, Raspberry Pi, and embedded systems. This work proposes a lightweight two-stage spatiotemporal pipeline that combines:

- ⚡ **YOLOv7-Tiny** — for fast, efficient per-frame spatial feature extraction and human/violence region detection
- 🔁 **Bi-GRU (Bidirectional Gated Recurrent Unit)** — for temporal sequence modeling across video frames in both forward and backward directions

The architecture achieves a strong balance between accuracy and computational efficiency, making it practical for real-time deployment on low-power hardware without requiring high-end GPUs.

---

## 🏆 Key Results

> ⚠️ **Note to repository owner:** Please fill in your actual metrics from the paper below.

| Metric        | Score          |
|---------------|----------------|
| Accuracy      | **XX.XX%**     |
| Precision     | **XX.XX%**     |
| Recall        | **XX.XX%**     |
| F1-Score      | **XX.XX%**     |
| FPS (Inference) | **XX FPS**  |
| Model Size    | **XX MB**      |

> Evaluated on: [RWF-2000 / Hockey Fight / Custom Dataset — update as applicable]

---

## 🏗️ Model Architecture

The pipeline is designed as a two-stage spatiotemporal system:

```
Input Video Stream
        │
   Frame Sampling
   (Sliding Window)
        │
 ┌──────▼──────┐
 │ YOLOv7-Tiny │  ← Spatial Stage
 │ (Per-Frame  │    Fast object detection
 │  Feature    │    + region proposals
 │  Extraction)│
 └──────┬──────┘
        │  Feature Vectors (frame sequence)
 ┌──────▼──────┐
 │   Bi-GRU    │  ← Temporal Stage
 │ (Forward +  │    Captures motion dynamics
 │  Backward   │    across time in both
 │  Sequence)  │    directions
 └──────┬──────┘
        │
  Dense Layer(s)
        │
  Binary Output
  ┌─────┴─────┐
  │  Violent  │  Non-Violent
  └───────────┘
```

**Why this combination?**
- YOLOv7-Tiny is significantly lighter than full YOLOv7, reducing GFLOPs while maintaining strong detection performance
- Bi-GRU avoids the gradient vanishing issue of LSTMs and is more computationally efficient (2 gates vs 4), while bidirectionality improves temporal context understanding
- Together they form a compact spatiotemporal pipeline ideal for edge deployment

---

## 📂 Datasets

> ⚠️ Please update this section with the specific datasets used in your paper.

The model was evaluated on standard benchmark violence detection datasets:

| Dataset          | Description                              | Videos   | Classes |
|------------------|------------------------------------------|----------|---------|
| RWF-2000         | Real-world fight surveillance footage    | 2,000    | 2       |
| Hockey Fight     | Fight scenes from hockey game footage    | 1,000    | 2       |
| [Custom Dataset] | [Describe any custom/additional data]    | XXXX     | 2       |

**Classes:** `Violence` / `Non-Violence`

---

## 🛠️ Requirements

```bash
Python >= 3.8
torch >= 1.12
torchvision
opencv-python
numpy
pandas
scikit-learn
matplotlib
tqdm
PyYAML
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

For YOLOv7-Tiny, clone the official YOLOv7 repository:

```bash
git clone https://github.com/WongKinYiu/yolov7.git
```

---

## 📁 Repository Structure

```
YOLOv7-Tiny-BiGRU-Violence-Detection/
│
├── data/
│   ├── train/
│   │   ├── violent/
│   │   └── non_violent/
│   ├── val/
│   └── test/
│
├── models/
│   ├── yolov7_tiny_extractor.py     # YOLOv7-Tiny feature extractor
│   ├── bigru_classifier.py          # Bi-GRU temporal classifier
│   └── pipeline.py                  # Full spatiotemporal pipeline
│
├── preprocessing/
│   ├── frame_sampler.py             # Video-to-frame sampling
│   └── feature_extractor.py         # Per-frame feature extraction
│
├── training/
│   ├── train.py                     # Training script
│   └── config.yaml                  # Hyperparameters and settings
│
├── evaluation/
│   ├── evaluate.py                  # Evaluation + metrics
│   └── confusion_matrix.py          # Visualization
│
├── inference/
│   ├── predict_video.py             # Real-time video inference
│   └── predict_stream.py            # Live CCTV stream inference
│
├── weights/
│   └── best_model.pth               # Trained model weights
│
├── results/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── training_curves.png
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

### 2. Prepare the Dataset

Organize your dataset in the following structure:

```
data/
├── train/
│   ├── violent/        ← violent video clips
│   └── non_violent/    ← non-violent video clips
├── val/
└── test/
```

### 3. Extract Frames

```bash
python preprocessing/frame_sampler.py --input data/train --output data/frames --fps 10
```

### 4. Train the Model

```bash
python training/train.py --config training/config.yaml --epochs 50 --batch_size 16
```

### 5. Evaluate the Model

```bash
python evaluation/evaluate.py --checkpoint weights/best_model.pth --data data/test
```

### 6. Run Inference on a Video File

```bash
python inference/predict_video.py --video path/to/video.mp4 --checkpoint weights/best_model.pth
```

### 7. Run on a Live Camera/CCTV Stream

```bash
python inference/predict_stream.py --source 0 --checkpoint weights/best_model.pth
```

---

## 📊 Ablation Study

> ⚠️ Please update this table with your actual ablation results.

| Configuration                    | Accuracy | FPS   | Model Size |
|----------------------------------|----------|-------|------------|
| YOLOv7 (full) + Bi-GRU          | XX.XX%   | XX    | XX MB      |
| YOLOv7-Tiny only (no temporal)  | XX.XX%   | XX    | XX MB      |
| YOLOv7-Tiny + GRU (unidirect.)  | XX.XX%   | XX    | XX MB      |
| **YOLOv7-Tiny + Bi-GRU (ours)** | **XX.XX%** | **XX** | **XX MB** |

---

## 🖥️ Deployment on Resource-Constrained Hardware

One of the key contributions of this work is demonstrating feasibility for edge deployment. The model was tested on:

> ⚠️ Please update with your actual hardware targets from the paper.

| Device              | FPS     | Status     |
|---------------------|---------|------------|
| NVIDIA GPU (XXXX)   | XX FPS  | ✅ Real-time |
| CPU Only            | XX FPS  | ✅ / ⚠️    |
| Raspberry Pi X      | XX FPS  | ✅ / ⚠️    |
| Edge Camera / IoT   | XX FPS  | ✅ / ⚠️    |

---

## 📈 Comparison with State-of-the-Art

> ⚠️ Please update with your actual comparison table from the paper.

| Method                    | Dataset     | Accuracy | Model Size |
|---------------------------|-------------|----------|------------|
| CNN-LSTM                  | RWF-2000    | XX.XX%   | Heavy      |
| 3D CNN                    | RWF-2000    | XX.XX%   | Heavy      |
| MobileNet-TSM             | RWF-2000    | 87.75%   | Light      |
| VGG + BiGRU               | RWF-2000    | XX.XX%   | Medium     |
| **YOLOv7-Tiny + Bi-GRU (Ours)** | RWF-2000 | **XX.XX%** | **Light** |

---

## 📖 Citation

If you find this work useful in your research, please cite:

```bibtex
@article{ejaz2026violence,
  title     = {Lightweight Temporal Violence Detection Using YOLOv7-Tiny and Bi-GRU for Resource-Constrained Environments},
  author    = {Md. Sabbir Ejaz and others},
  journal   = {Smart Cities and Society},
  publisher = {Elsevier},
  year      = {2026},
  doi       = {10.1016/j.scs.2026.XXXXXX},
  url       = {https://www.sciencedirect.com/science/article/pii/S2772941926000578}
}
```

> ⚠️ Please verify and complete the author list, volume, issue, and DOI from your published paper before finalizing.

---

## 🔮 Future Work

Planned directions for extending this work:

- [ ] Multi-class violence classification (e.g., fighting, robbery, vandalism)
- [ ] Integration with real-time alert/notification systems
- [ ] Further model compression via quantization and pruning (<5 MB target)
- [ ] Cloud-based deployment (AWS/GCP/Azure) with RTSP stream support
- [ ] Multi-camera scene understanding and tracking
- [ ] Transformer-based temporal modeling (Video Swin Transformer) comparison
- [ ] Dataset expansion with diverse real-world CCTV footage

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- The open-source communities behind [YOLOv7](https://github.com/WongKinYiu/yolov7) and PyTorch
- Benchmark dataset creators: RWF-2000, Hockey Fight
- Surveillance and public safety research communities

---

<p align="center">Made with ❤️ for safer public environments</p>

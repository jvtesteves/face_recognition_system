# Face Recognition System

A complete real-time face detection and recognition system built from scratch with Python, OpenCV and `facenet-pytorch`. Features:

- Multi-face detection in live video (webcam)  
- Automatic face crop capture for training  
- Embedding extraction using a pretrained FaceNet (InceptionResnetV1)  
- Real-time recognition by comparing embeddings against a template database  
- “Known” vs. “Unknown” classification with adjustable threshold  

---

## 📁 Directory Structure

```
face_recognition_system/
│
├── dataset/                  # Generated at runtime (not committed)
│   └── joao/                 # Training images per identity
│
├── detectors/                # (Optional) Custom detection modules
├── embeddings/               # (Optional) Custom embedding/extraction modules
├── utils/                    # (Optional) Helper functions (logging, I/O, etc.)
│
├── .gitignore                # Ignores venv, dataset, embeddings, caches
├── requirements.txt          # Project dependencies
├── capture_faces.py          # Capture face crops from original frame
├── detect_faces_mtcnn.py     # Real-time detection test with MTCNN
├── extract_embeddings.py     # Extracts and saves embeddings to `embeddings.pt`
├── recognize_faces.py        # Real-time recognition using stored templates
└── README.md                 # This file
```

---

## ⚙️ Prerequisites

- Python 3.8+  
- Git  
- Integrated or USB webcam  
- (Optional) CUDA-enabled GPU for faster inference  

### Dependencies

```bash
pip install -r requirements.txt
```

Your `requirements.txt` might include:

```text
opencv-python
torch
torchvision
facenet-pytorch
pillow
```

---

## 🚀 Installation Guide

1. **Clone the repo**  
   ```bash
   git clone https://github.com/your-username/face_recognition_system.git
   cd face_recognition_system
   ```

2. **Create and activate a virtual environment**  
   ```bash
   python -m venv .venv
   source .venv/bin/activate    # macOS/Linux
   .venv\Scripts\activate       # Windows
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **macOS SSL certificates** (if you see SSL errors when downloading weights)  
   - Run **Install Certificates.command** located in `/Applications/Python 3.x/`.

---

## 🛠️ Usage

### 1. Test MTCNN detection

```bash
python detect_faces_mtcnn.py
```

Opens your webcam and draws green boxes around all detected faces.

---

### 2. Capture training images

```bash
python capture_faces.py
```

- Captures 20 face crops (160×160) from the webcam into `dataset/<identity>/`.  
- Press `ESC` to stop early.

---

### 3. Extract embeddings

```bash
python extract_embeddings.py
```

- Loads images from `dataset/` and extracts 512-dim embeddings with FaceNet.  
- Saves a dictionary of tensors to `embeddings.pt`.

---

### 4. Real-time recognition

```bash
python recognize_faces.py
```

- Loads `embeddings.pt`, computes the average template per identity.  
- Detects faces, extracts embeddings and compares to templates.  
- Displays identity (green) or “Unknown” (red) with distance score.

**Sensitivity threshold**  
Edit at the top of `recognize_faces.py`:

```python
threshold = 0.8  # 0.6 = stricter, 1.0 = more lenient
```

---

## 🔧 How to Add New Identities

1. Change `person_name` in `capture_faces.py` (and create `dataset/<name>/`).  
2. Run the script to capture faces for the new identity.  
3. Run `extract_embeddings.py` again to update `embeddings.pt`.  
4. Now `recognize_faces.py` will recognize the new name as well.

---

## 💡 Best Practices & Tips

- **Alignment**: use facial landmarks (eyes, nose) to align and normalize faces before embedding.  
- **Cosine similarity**: swap Euclidean distance for `F.cosine_similarity()` for greater robustness.  
- **Logging**: record recognition events to CSV for later analysis.  
- **GUI**: consider Streamlit or Flask for a simple web interface.

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

1. Fork this repository.  
2. Create a feature branch: `git checkout -b my-feature`.  
3. Commit your changes: `git commit -m "Add my feature"`.  
4. Push to the branch: `git push origin my-feature`.  
5. Open a Pull Request.

---

## 📫 Contact

João Victor Tavares Esteves  
✉️ [joaovtesteves2002@gmail.com](mailto:joaovtesteves2002@gmail.com)  
🌐 [GitHub](https://github.com/jvtesteves)  

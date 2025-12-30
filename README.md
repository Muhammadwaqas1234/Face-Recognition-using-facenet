# Face Verification System using FaceNet & Siamese Network

This project implements a **Face Verification system** using a **pretrained FaceNet model** and a **Siamese Network architecture**.  
The system determines whether **two face images belong to the same person or not** by measuring the distance between their embeddings.

The project is built using **TensorFlow/Keras** and evaluated on the **LFW (Labeled Faces in the Wild)** dataset.

---

## 🔍 What is Face Verification?
Face verification answers a simple question:

> **“Do these two face images belong to the same person?”**

This is different from face recognition (classification), where the goal is to identify *who* the person is.

---

## 🚀 Key Features
- ✅ Uses **pretrained FaceNet** (no training from scratch)
- ✅ **Siamese Network** for similarity learning
- ✅ **Contrastive Loss**
- ✅ Automatic **threshold tuning**
- ✅ Evaluation using **Accuracy, ROC Curve, and EER**
- ✅ Multiple **visualizations** for analysis
- ✅ Clean and modular TensorFlow code
- ✅ Ready to run on **Google Colab**

---

## 🧠 Model Architecture (Concept)
```

Image 1 ─┐
├─ FaceNet ─→ Embedding 1 ─┐
Image 2 ─┘                          ├─ L2 Distance → Similarity Score
└─ FaceNet ─→ Embedding 2 ─┘

````

- Same FaceNet model is shared between both inputs
- Output is a **distance value**
- Small distance → Same person
- Large distance → Different persons

---

## 📊 Dataset
**LFW (Labeled Faces in the Wild)**  
- Real-world face images
- Multiple identities
- Challenging lighting, pose, and expressions

Downloaded automatically using Kaggle.

---

## 🛠️ Tech Stack
- Python 3
- TensorFlow / Keras
- keras-facenet
- NumPy
- scikit-learn
- Matplotlib & Seaborn
- Google Colab

---

## ⚙️ Installation

```bash
pip install tensorflow keras-facenet scikit-learn matplotlib seaborn kaggle
````

---

## ▶️ How to Run

1. Upload `kaggle.json` to Colab
2. Run dataset download cells
3. Generate face pairs
4. Load FaceNet model
5. Build Siamese network
6. Tune threshold
7. Evaluate on test set
8. Visualize results

---

## 📈 Evaluation Metrics

* **Verification Accuracy**
* **ROC Curve & AUC**
* **Equal Error Rate (EER)**
* **Confusion Matrix**
* **Distance Distributions**
* **t-SNE Embedding Visualization**

---

## 📊 Example Results (Typical)

| Metric   | Value     |
| -------- | --------- |
| Accuracy | 93,5% |
| AUC      | ≥ 0.98    |
| EER      | ≤ 6%     |

*(Results depend on pair generation and threshold tuning)*

---

## 📌 Important Notes

* FaceNet is used as a **frozen feature extractor**
* No backbone retraining (best practice)
* Threshold selection is **critical**
* Siamese model performs **verification**, not classification

---

## 🧪 Visualizations Included

* Distance distribution plots
* ROC curve
* Precision-Recall curve
* Confusion matrix
* False positive / false negative face pairs
* t-SNE visualization of embeddings

---

## 🔮 Future Improvements

* Triplet Loss with hard negative mining
* ArcFace / InsightFace integration
* Real-time webcam verification
* Mobile optimization (MobileFaceNet)
* Face alignment and augmentation

---

## 📚 References

* FaceNet: A Unified Embedding for Face Recognition and Clustering
* LFW Dataset
* TensorFlow & Keras Documentation

---

## 👤 Author

**Muhammad Waqas**
Machine Learning / Computer Vision Enthusiast


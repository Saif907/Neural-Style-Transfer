# 🎨 Neural Style Transfer with Transformer Network

A fast and efficient neural style transfer project that applies artistic styles to images using a Transformer-based deep learning architecture. Designed for real-time performance, the model incorporates **depthwise-separable convolutions**, **residual connections**, **SE attention**, and **VGG-based perceptual loss** for high-quality results.

> ✅ Trained on COCO 2014 and tested on an RTX 3050 (6GB) GPU for efficient, real-time inference.

---

## 📌 Key Features

- 🔁 Two model versions:
  - **v1**: Custom Transformer-based architecture with SE attention.
  - **v2**: Based on *Johnson et al.* (CVPR 2016) with improvements.
- 🧠 VGG-16-based perceptual loss for content-aware style transfer.
- ⚡ Fast inference using lightweight, optimized layers.
- 🖼️ Supports multiple trained styles (e.g., *Van Gogh*, *Mosaic*).
- 🧰 Modular and easy-to-extend codebase.

---

## 🧪 Skills & Technologies

**Python · PyTorch · Deep Learning · Computer Vision · CNN · Transformer Networks · Style Transfer · Perceptual Loss · GPU Acceleration**

---

## 🗂️ Project Structure

```

NeuralStyleTransfer/
├── data/               # Content and style images
│   ├── content/
│   └── styles/
│
├── outputs/
│   ├── samples/        # Stylized outputs
│   └── before\_after/   # Before and after comparisons
│
├── saved\_models/       # Trained models (e.g., van\_gogh.pth)
│
├── models/
│   ├── transformer\_netv1.py
│   ├── transformer\_netv2.py
│   └── vgg.py
│
├── scripts/
│   ├── train.py
│   ├── app.py
│   └── utils.py
│
├── requirements.txt
├── config.yaml         # Optional: configuration for training/inference
└── README.md

````

---

## 🚀 How to Run

### 🔧 Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/NeuralStyleTransfer.git
   cd NeuralStyleTransfer
````

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

### 🏋️‍♀️ Train a Model

```bash
python scripts/train.py \
  --content-dir data/content \
  --style-image data/styles/van_gogh.jpg \
  --save-model saved_models/van_gogh.pth
```

---

### 🖼️ Stylize an Image

```bash
python scripts/app.py \
  --model saved_models/van_gogh.pth \
  --input data/content/example.jpg \
  --output outputs/samples/stylized.jpg
```

---

## 📊 Examples

| Original                                   | Stylized (Van Gogh)                      |
| ------------------------------------------ | ---------------------------------------- |
| ![Before](outputs/before_after/before.jpg) | ![After](outputs/before_after/after.jpg) |

> More examples available in the `outputs/` folder.

---

## 🧠 Future Improvements

* 🎥 Add support for real-time **video style transfer**
* 🌐 Build a user-friendly **web UI** (e.g., Streamlit or Flask)
* 🧪 Explore advanced **style loss techniques**

---

## 📚 Reference

* Johnson, Justin, Alexandre Alahi, and Li Fei-Fei.
  [*Perceptual Losses for Real-Time Style Transfer and Super-Resolution*](https://arxiv.org/abs/1603.08155), CVPR 2016.

---

## 🤝 Connect

**Author:** Saif Shaikh
📫 Email: [Gmail](mailto:saif81868@gmail.com)
🔗 LinkedIn: [linkedin.com/in/saif-shaikh-527346251](https://www.linkedin.com/in/saif-shaikh-527346251)

```

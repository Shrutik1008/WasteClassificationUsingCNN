# 🧠 WasteClassificationCNN

An AICTE Virtual Internship (Cycle 3) project focused on **waste classification using Convolutional Neural Networks (CNN)**.  
This project helps identify and categorize waste into different types using image classification techniques, contributing towards environmental sustainability.

---

## 📁 Dataset

The dataset used in this project is available on [Kaggle](https://www.kaggle.com/datasets/techsash/waste-classification-data).

You can also download it programmatically using:

```python
import kagglehub  # You may need to run: pip install kagglehub

path = kagglehub.dataset_download("techsash/waste-classification-data")
print("Path to dataset files:", path)
```

---

## 🛠️ Technologies Used

- Python 🐍
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- Jupyter Notebook
- Kaggle API

---

## 🧠 Model Overview

We use a CNN (Convolutional Neural Network) to classify waste images into categories like:

- Organic
- Recyclable
- Hazardous
- General Waste

The model is trained on labeled images and evaluated using accuracy, loss, and confusion matrix.

---

## 📷 Screenshots

### 📌 Dataset Sample
![Dataset Sample](assets/screenshot1.png)

### 🧠 Model Architecture
![Model Architecture](assets/screenshot2.png)

### 📈 Training Results
![Training Results](assets/screenshot3.png)

---

## 🚀 Getting Started

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/WasteClassificationCNN.git
   cd WasteClassificationCNN
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Dataset**
   Use the Kaggle link above or download via `kagglehub`.

4. **Run the Project**
   Launch the notebook or script to start training:
   ```bash
   jupyter notebook WasteClassificationCNN.ipynb
   ```

---

## 📌 Future Improvements

- Improve classification accuracy with deeper architectures.
- Add data augmentation and transfer learning.
- Deploy model with a user interface using Streamlit or Flask.

---

## 🙏 Acknowledgements

- [AICTE Virtual Internship](https://www.aicte-india.org/)
- [TechSaksham - Microsoft & SAP Initiative](https://www.techsaksham.org/)
- [Kaggle Dataset by TechSash](https://www.kaggle.com/datasets/techsash/waste-classification-data)

---

# EuroSAT Land Use Classification

## 📌 Project Overview
This project classifies satellite images into different land-use categories using deep learning. The model is trained on the **EuroSAT dataset**, which consists of multispectral satellite images captured by the **Sentinel-2 satellite**. It helps in **land monitoring, urban planning, environmental protection, and disaster management**.

---

## 📂 Dataset Details
- **Name:** EuroSAT (Sentinel-2) Dataset
- **Source:** [EuroSAT Dataset on Kaggle](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset-27500-images-10-classes)
- **Total Images:** 27,500
- **Image Size:** 64×64 pixels (RGB)
- **Classes (10 Categories):**
  - 🌿 **AnnualCrop**
  - 🌳 **Forest**
  - 🌾 **HerbaceousVegetation**
  - 🛣 **Highway**
  - 🏭 **Industrial**
  - 🐄 **Pasture**
  - 🌱 **PermanentCrop**
  - 🏡 **Residential**
  - 🌊 **River**
  - 🌊 **SeaLake**

---

## 🎯 Problem Statement
**Objective:** Develop a deep learning model to classify satellite images into different land-use categories. This helps in **urban planning, environmental monitoring, and disaster response** by automating land classification from satellite data.

---

## 🏗 Model Details
- **Architecture:** Convolutional Neural Network (CNN)
- **Framework:** TensorFlow/Keras
- **Saved Model File:** `eurosat_model2.keras`
- **Input Shape:** (64, 64, 3)
- **Output:** 10-class classification

---

## 🚀 Usage
1. **Run the Flask App:**
   ```sh
   python app.py
   ```
2. **Upload an Image:** Navigate to `http://127.0.0.1:5000/` and upload a satellite image for classification.
3. **View Prediction:** The app displays the predicted land-use class.

---

## 🔗 References
- **EuroSAT Paper:** [https://arxiv.org/abs/1709.00029](https://arxiv.org/abs/1709.00029)
- **Dataset Source:** [EuroSAT on Kaggle](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset-27500-images-10-classes)

---

**💡 Future Improvements:**
- Implement **real-time satellite image analysis** using APIs.
- Integrate **GIS-based visualization**.
- Deploy as a **web-based land monitoring system**.


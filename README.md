# 🧠 MCI-to-AD MRI Classification

This project uses **EfficientNetB0 Transfer Learning** to classify MRI brain scans into 4 stages of cognitive decline:

- **EMCI** (Early Mild Cognitive Impairment)
- **LMCI** (Late Mild Cognitive Impairment)
- **MCI** (Mild Cognitive Impairment)
- **AD** (Alzheimer's Disease)

## 📂 Dataset
- Custom MRI dataset with 4 labeled classes.
- Split into training & validation using `ImageDataGenerator`.

## 🏗 Model
- **Base model:** EfficientNetB0 (ImageNet weights)
- GlobalAveragePooling + Dropout + Dense(4, softmax)

## 📊 Training
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Validation split: 20%

## 🚀 Results
- Shows per-class accuracy & confusion matrix
- Predicts patient’s stage of cognitive impairment

## 📦 How to run
```bash
pip install -r requirements.txt
python train.py


## 🔮 Future Work
-Add Grad-CAM visualization to see MRI regions that influence the model.

-Experiment with deeper models like EfficientNetB3.
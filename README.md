# Sentinel-Net
Sentinel-Net for Anomaly Events Detection of Meteorological Sta-tions Monitoring
## ⚙️ Environment Setup
Install required dependencies:
bash
pip install -r requirements.txt
## 📦 Dataset Preparation
You can access this dataset from the provided https://pan.baidu.com/s/1lE1ffEIvluwZwNtztmeN9w?pwd=n74t.

normal/: normal samples

anomaly/: abnormal samples

Supported formats: .png or .jpg.

## 🧠 Train and Test the Model
Run:

bash python model/test_sentinel.py
The script will prompt for one of the following operations:

Enter train: to train a new model

Enter load: to load an existing model and perform inference

## 🔁 Train a New Model
Choose an activation function (relu, leaky_relu, elu, prelu, swish)

Model will be trained on normal data

Trained model is saved to output/model_attention.h5

Training loss is saved as loss_data_sentinel.csv, and a loss curve will be generated

## 📊 Load and Run Inference
Automatically loads the latest model (or specify a path manually)

Performs inference on both normal and anomaly samples in the val set

Outputs inference time, reconstruction results, loss, and accuracy

### 🧾 Output Files
Model: output/<timestamp>/model_attention.h5

Loss data: output/loss_data_sentinel.csv

Inference results: printed in terminal (includes timing, loss, and accuracy)

## ⚠️ Notes
Ensure the font file (e.g., simhei.ttf) path is correct for displaying Chinese characters

Modify paths in test_sentinel.py to match your dataset

Supported image formats: .png and .jpg

## 📬 Contact
For any issues or questions, please contact the project maintainer.
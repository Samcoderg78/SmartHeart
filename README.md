# SmartHeart: AI-Powered Heart Disease Risk Analyzer ❤️

**SmartHeart** is an interactive, explainable AI web app that lets users:

* 🧠 Predict their 10-year cardiovascular (CVD) risk using the **Framingham Heart Study** model, enhanced with machine learning
* 🔄 Explore “what if” lifestyle scenarios and see how their risk would change (with **SHAP** explanations)
* 📝 Instantly generate and download a **personalized health report card** as a PDF
* 📈 Track and visualize their CVD risk over time
* 🌐 Use the platform in their **preferred language** (multilingual support)

---

## 🚀 Quickstart (Using Docker)

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/smartheart.git
cd smartheart
```

### 2. Place the Framingham Dataset

* Download `framingham.csv` from [Kaggle](https://www.kaggle.com/datasets).
* Place it inside the `data/` directory:

```bash
mkdir -p data
mv framingham.csv data/framingham.csv
```

### 3. Build the Docker Image

```bash
docker build -t smartheartapp .
```

### 4. Run the App

```bash
docker run -p 8501:8501 smartheartapp
```

Open your browser and navigate to [http://localhost:8501](http://localhost:8501)

---

## 🛠️ Developer: Local Setup

```bash
# (Recommended: create a virtual environment)
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧪 Model Training (Optional: If You Want to Retrain)

Train inside Docker for full reproducibility:

```bash
docker run -it --rm -v "${PWD}:/app" smartheartapp /bin/bash
python models/train_model.py
exit
```

The trained model will be saved to:
`models/saved_models/ensemble_model.pkl`
Rebuild the Docker image if needed.

---

## ✨ Features

* **🧮 Risk Calculator**: Predicts 10-year CHD risk using clinical & lifestyle data
* **🔧 Lifestyle Simulator**: Simulate effects of quitting smoking, weight loss, etc.
* **📊 Explainable AI**: SHAP visualizations show personal risk factors
* **📄 Report Card**: Download a personalized PDF for health records
* **📆 Time-Series Tracker**: Track risk changes and export history as CSV
* **🌍 Multilingual Support**: English, Japanese, Spanish, French, Chinese (easily extendable)

---

## 📁 Directory Structure

```
.
├── app.py
├── components/
│   ├── sidebar.py
│   ├── risk_calculator.py
│   ├── lifestyle_simulator.py
│   ├── report_card.py
│   └── time_series_tracker.py
├── models/
│   ├── risk_model.py
│   ├── train_model.py
│   └── saved_models/
│       └── ensemble_model.pkl
├── utils/
│   ├── explainer.py
│   └── translator.py
├── data/
│   └── framingham.csv
├── requirements.txt
├── requirements.docker.txt
├── Dockerfile
└── README.md
```

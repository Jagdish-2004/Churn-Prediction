# Customer Churn Prediction using ANN

## 📌 Project Overview
This project demonstrates the use of **Artificial Neural Networks (ANN)** to predict **customer churn** in a banking dataset. The workflow includes:
- Data preprocessing  
- Feature encoding (Label Encoding & One-Hot Encoding)  
- Feature scaling  
- Baseline machine learning models  
- ANN model training and evaluation  
- Deployment via a **Streamlit app** (`app.py`)  

---

## 📂 Project Structure
```
├── ANN.ipynb              # Jupyter Notebook containing training workflow
├── ann_model.h5           # Trained ANN model
├── label_encoder.pkl      # Saved label encoder
├── onehot_encoder.pkl     # Saved one-hot encoder
├── scaler.pkl             # Saved standard scaler
├── app.py                 # Streamlit deployment app
```

---

## ⚙️ Installation & Requirements
Install dependencies using:

```bash
pip install -r requirements.txt
```

### Main Libraries Used
- **pandas** – Data handling  
- **numpy** – Numerical computations  
- **matplotlib / seaborn** – Visualization  
- **scikit-learn** – Preprocessing & baseline models  
- **TensorFlow / Keras** – ANN training  
- **pickle** – Saving encoders/scalers  
- **streamlit** – Deployment  

---

## 🚀 Running the Notebook
1. Open `ANN.ipynb` in Jupyter.  
2. Run all cells to preprocess, train, and evaluate.  
3. Trained ANN and preprocessing objects (`.h5`, `.pkl`) will be saved.  

---

## 🌐 Running the Streamlit App
To launch the deployment app:

```bash
streamlit run app.py
```

This opens an interactive web UI where users can input customer details and get churn predictions.  

---

## 📊 Model Workflow
1. **Preprocessing**
   - Encode categorical variables (`Gender`, `Geography`)  
   - Scale numerical features  
2. **Baseline Models**
   - Logistic Regression, Decision Tree, Random Forest  
3. **ANN Model**
   - Architecture: Dense(64, relu) → Dense(32, relu) → Dense(1, sigmoid)  
   - Loss: Binary Crossentropy, Optimizer: Adam  
4. **Evaluation**
   - Accuracy, Precision, Recall, F1-score  

---

## 📈 Results
- ANN achieved higher predictive performance than classical ML models.  
- Streamlit app enables real-time churn prediction.  

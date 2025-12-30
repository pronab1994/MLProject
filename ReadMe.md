<p align="center">
  <img src="assets/ml_pipeline.jpg" alt="Machine Learning Pipeline" width="85%">
</p>

### Pipeline Flow
1. **Data Ingestion** – Reads and validates raw data  
2. **Data Transformation** – Feature engineering & preprocessing  
3. **Model Training** – Trains and evaluates the ML model  
4. **Artifact Generation** – Saves trained artifacts  
5. **Prediction Pipeline** – Loads artifacts for inference  

This design ensures **training and inference are fully decoupled**, improving reusability and production readiness.

---

## 🌐 Web Application Architecture (Unified Flask App)

### How the Web App Works
1. User enters details in the UI form (`templates/index.html`)  
2. Flask handles the request (`application.py`)  
3. Inputs are validated and structured into a schema  
4. `PredictPipeline` loads trained artifacts:
   - `artifacts/preprocessor.pkl`
   - `artifacts/model.pkl`
5. Model generates prediction  
6. Result is rendered under **Prediction Result** in the UI  

📌 **Backend entry point:** `application.py`

---

## 🧪 How to Run Locally

```bash
git clone <your-repo-url>
cd MLProject
pip install -r requirements.txt
python src/pipeline/train_pipeline.py
python application.py

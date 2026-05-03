```markdown
#  Churn Prediction System (V2 - Production Ready)

A complete end-to-end Machine Learning system for predicting customer churn.  
This project demonstrates how to transform raw customer data into a production-ready ML service with feature engineering, training pipeline, evaluation layer, and REST API deployment.

---

#  Business Problem

Companies lose customers without understanding why.  
This system predicts whether a customer is likely to churn, enabling:

- Early intervention strategies  
- Targeted retention campaigns  
- Revenue protection and optimization  

---

#  Solution Overview

This project implements a full ML pipeline:

-  Feature engineering from raw behavioral data  
-  Supervised ML model (RandomForestClassifier)  
-  End-to-end training pipeline  
-  Model evaluation (Precision / Recall / F1)  
-  FastAPI deployment for real-time inference  
-  Docker support for production deployment  
-  Logging system for observability  

---

#  System Architecture

data/raw → preprocessing → feature engineering → training → evaluation → model saving  
                                                     ↓  
                                            FastAPI service  
                                                     ↓  
                                        real-time predictions  

---

#  Project Structure

```bash

churn_system/
│
├── data/
├── models/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── services/
│   ├── api/
│   ├── evaluation/
│   └── utils/
│
├── train_pipeline.py
├── test_service.py
├── evaluate_model.py
├── Dockerfile
└── requirements.txt

````
---

#  Installation

```bash
git clone <repo-url>
cd churn_system

python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
````

---

#  Run Training Pipeline

```bash
python train_pipeline.py
```

This will:

* Load data
* Build features
* Train model
* Evaluate performance
* Save artifacts (`model.pkl`, `scaler.pkl`, `feature_config.pkl`)

---

#  Run API

```bash
uvicorn src.api.main:app --reload
```

Open:

👉 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

#  API Endpoint

## POST `/classify`

### Request

```json
{
  "age": 35,
  "last_interaction": 12,
  "num_sessions": 20,
  "purchase_frequency": 3,
  "support_tickets": 1,
  "tenure_months": 24
}
```

### Response

```json
{
  "churn_probability": 0.78,
  "risk": "HIGH"
}
```

---

#  Model Evaluation

Metrics used:

* Precision
* Recall
* F1-score

This ensures proper handling of class imbalance (critical for churn problems).

---

#  Key Features

* Feature engineering layer (behavioral signals)
* Model + scaler persistence
* Feature consistency via `feature_config.pkl`
* Logging system for observability
* Modular architecture (production-grade design)

---

#  Docker Deployment

```bash
docker build -t churn-system .
docker run -p 8000:8000 churn-system
```

---

#  Business Impact

* Reduce customer churn rate
* Improve customer retention strategies
* Enable data-driven decision making
* Integrate into CRM / dashboards

---

#  Author

**Amine Jahed**
Self-taught ML / GenAI Engineer

Focus:

* Machine Learning Systems Design
* Production AI APIs
* GenAI & RAG Systems

---

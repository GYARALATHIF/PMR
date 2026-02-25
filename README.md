# Analysis-of-Patient-Mortality-and-Readmission-using-MIMIC-iii-Dataset

## Live Web App (Production Version)

This repository now includes a full-stack real-time web app:

- `backend/main.py`: FastAPI backend (REST + WebSocket)
- `templates/index.html`: responsive browser UI
- `static/`: frontend JavaScript and styles
- `render.yaml`, `Procfile`, `Dockerfile`: deployment support

### Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

Open `http://127.0.0.1:8000`.

### Publish live (Render)

1. Push this project to your GitHub repository.
2. Go to Render and create a new **Web Service** from that repo.
3. Render will auto-detect `render.yaml` (or use):
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
4. Keep `USE_BIOCLINICAL_BERT=false` for faster + cheaper deploy startup.
5. Deploy and open your live URL.

### Optional ClinicalBERT mode

Set `USE_BIOCLINICAL_BERT=true` if you want real transformer embeddings at runtime.  
If disabled, the app uses a deterministic 768-dim fallback embedding so predictions continue working.

[Abstract](#Abstract) 

[Contents](#Contents)

[1 Introduction](#1-introduction)  
[1.1 Problem Statement](#11-problem-statement)  
[1.2 Objectives](#12-objectives)  
[1.3 Societal Applications](#13-societal-applications)  

[2 Literature Survey](#2-literature-survey)  
[2.1 Introduction](#21-introduction)  
[2.2 Analyzing the Existing System](#22-analyzing-the-existing-system)  

[3 Dataset Description](#3-Dataset-Description)  
[3.1 Overview](3.1-overview)  
[3.2 Data Cleaning and Preprocessing](3.2-Data-Cleaning-and-Preprocessing)

[4 Analysis](#4-analysis)  
[4.1 Visualizations](#4-visualizations)  

[5 Feature Engineering](#5-Feature-Engineering)

[6 Model Development](#6-model-development)  
[6.1. Objective](#6.1.-Objective)  
[6.2. Models Implemented](#6.2.-Models-Implemented)  
[6.3. Evaluation Metrics](#6.3.-Evaluation-Metrics)  
[6.4. Visualizations](#6.4.-Visualizations)  
[6.5. Model Interpretability Using SHAP](#6.5.-Model-Interpretability-Using-SHAP)  
[6.6. Preprocessing and Deployment Pipeline](#6.6.-Preprocessing-and-Deployment-Pipeline)  
[6.7. Known Issues](#6.7.-Known-Issues)  
[6.8. Technologies Used](#6.8.-Technologies-Used)  
[6.9. Future Work](#6.9.-Future-Work)  
[6.10. Installation Requirements](#6.10.-Installation-Requirements)  
[6.11. Features Used](#6.11.-Features-Used)  
[6.12. Model Results Summary](#6.12.-Model-Results-Summary)  
[6.13. Best Model](#6.13.-Best-Model)  

[7 Results](#7-results)  

[8 Conclusion](#8-Conclusion)  
[8.1 Future Scope](#8.1-Future-Scope)


[9 References](#9-References)

# Abstract

We propose a multimodal predictive modeling framework aimed at enhancing the prediction of in-hospital mortality and hospital readmission by integrating both structured and unstructured clinical data from the MIMIC-III dataset. Our approach combines quantitative features such as diagnoses, procedures, and demographic information with narrative insights derived from clinical notes using ClinicalBERT embeddings. This fusion allows us to capture a more holistic view of patient health by leveraging both numeric trends and contextual language patterns in clinical documentation.

To strengthen the temporal understanding of patient trajectories, we incorporate time-stamped clinical events and analyze how the progression of medical conditions impacts outcomes. The model is benchmarked against established baselines including GRU-D and BERT-only architectures to assess its predictive performance. Additionally, we utilize interpretable machine learning techniques such as SHAP to gain insights into model behavior and feature influence.

Ultimately, our goal is to build a robust, interpretable, and scalable system that supports clinicians in identifying high-risk patients early, thereby improving care delivery, reducing readmission rates, and saving lives.

# Contents

## 1 Introduction

In today’s data-driven healthcare era, technology is transforming how we understand and respond to patient needs. Hospitals around the world are moving toward smarter, more proactive care models, where predicting a patient’s health trajectory is just as important as treating the present illness. Imagine a system that could warn doctors, “This patient is at high risk of being readmitted,” or alert nurses with insights like “This individual’s vital signs suggest a higher risk of mortality.” That’s the power of predictive modeling in healthcare.

Just like virtual assistants have revolutionized the way we handle daily tasks by understanding speech and making intelligent decisions, similar intelligence can be harnessed to support clinical decision-making. Hospital readmissions and in-hospital mortality are critical indicators of healthcare quality and resource utilization. Predicting these outcomes ahead of time could save lives, reduce healthcare costs, and allow for more personalized and preventive care.

With access to massive repositories of clinical data such as MIMIC-III and advancements in machine learning and natural language processing, especially using models like ClinicalBERT, we can now mine valuable insights from both structured data (like lab results, vitals, diagnoses) and unstructured data (like doctor’s notes). These insights enable building intelligent systems that can forecast patient outcomes, guide clinical interventions, and ultimately improve the quality of care. This project builds upon that vision—leveraging openly available clinical datasets and cutting-edge AI to build a virtual assistant for hospitals, capable of predicting readmissions and mortality risk, and supporting smarter, data-driven decisions at the bedside.

## 1.1 Problem Statement

Every year, thousands of patients are readmitted to hospitals within a short time after discharge, and many face increased risk of mortality due to undetected complications or lack of timely intervention. Despite advances in healthcare, predicting such outcomes remains a major challenge for hospitals and care providers. Doctors and nurses are often overwhelmed with large volumes of patient data, making it difficult to manually assess risks or anticipate deteriorations in real-time.

The main agenda of this project is to solve this problem. The solution is to build an intelligent assistant for hospitals that can automatically predict the likelihood of patient readmission and mortality based on both clinical records and physician notes. By using advanced machine learning models and natural language processing techniques, this system can support healthcare professionals in making informed decisions and improve patient outcomes.

## 1.2 Objectives

- The main objective of building a hospital readmission and mortality prediction system is to assist healthcare professionals in identifying high-risk patients early.

- Another objective is to reduce preventable hospital readmissions and improve patient outcomes by enabling timely interventions.

- Fundamental objectives that maximize the overall value of predictive healthcare systems are accuracy, reliability, and clinical relevance.

- Design a user-friendly and interpretable system that can be integrated into existing hospital workflows with minimal disruption.

- Improve efficiency in healthcare decision-making by automating the analysis of large-scale patient data using machine learning.

- Enhance the use of both structured clinical data and unstructured physician notes through advanced natural language processing.

- Provide actionable insights to doctors and caregivers, helping them prioritize care, manage resources better, and ultimately save lives.

## 1.3 Societal Applications

- This application has significant implications for society, especially in improving healthcare delivery and saving lives through proactive medical intervention.

- This system can help hospitals prioritize high-risk patients, reducing preventable deaths and enhancing overall patient care.

- It can support overburdened healthcare systems by streamlining decision-making and reducing manual workload for medical staff.

- This project is feasible for future upgrades, including integration with electronic health records and real-time alert systems for critical care.

- It enhances healthcare equity by using data to identify at-risk individuals who might otherwise be overlooked.

- It provides timely insights that enable better discharge planning, reducing the emotional and financial burden on patients and families.

- By automating predictions of adverse outcomes, it allows physicians to focus more on treatment and human interaction rather than data crunching.

- It ensures continuous data monitoring and analysis, helping institutions stay prepared and responsive to emergencies.

- Even with emerging challenges in healthcare, this intelligent system supports consistent, evidence-based decision-making and better resource allocation.

# 2 Literature Survey

## 2.1 Introduction

This section presents a review of existing research and systems developed for predicting hospital readmissions and patient mortality. Numerous studies and healthcare analytics solutions have been proposed that leverage data-driven models and machine learning to address these critical issues. These systems typically focus on analyzing patient data to identify those at high risk of readmission or death, enabling timely interventions and better resource allocation.

## 2.2 Analyzing the Existing System

Many existing systems apply statistical methods and machine learning algorithms to electronic health records (EHRs) for risk prediction. Popular models include Logistic Regression, Decision Trees, Random Forests, and Support Vector Machines (SVMs). More recent approaches use deep learning methods like LSTM and CNNs to capture temporal patterns in clinical data.

Hospitals and research institutions utilize tools such as Python, R, and platforms like TensorFlow or Scikit-learn to build predictive models. Additionally, frameworks like Apache Spark are used to handle large-scale healthcare datasets.

One widely known model is the LACE index (Length of stay, Acuity of admission, Comorbidities, and Emergency visits), which helps estimate the risk of readmission. Other institutions use proprietary scoring systems integrated into their EHR systems.

These predictive systems have shown success in identifying at-risk patients, but challenges such as data imbalance, missing values, and interpretability of black-box models remain. Ensuring model fairness and privacy compliance is also essential in clinical applications.

# 3 Dataset Description

The dataset used in this project is derived from MIMIC-III v1.4, a publicly available critical care database. It consists of multiple tables that provide comprehensive information about patients, their hospital admissions, diagnoses, treatments, and medications. The data spans over 40,000 hospital admissions, offering valuable insights into patient care, outcomes, and hospital performance.

## 3.1 Overview

The dataset consists of 5 main tables, each providing a specific type of information about patients and their hospital stay. These datasets allow for a deep analysis of patient demographics, admission details, medical history (diagnoses and procedures), and treatments (medications).

The data enables a comprehensive approach to understanding patient outcomes, including mortality, readmission, and the impact of various diagnoses, treatments, and patient characteristics. This structured information is essential for building predictive models in healthcare analytics, such as predicting patient outcomes or identifying risk factors for readmission.

## Dataset Files

The project utilizes 5 key datasets from the MIMIC-III v1.4 database, each contributing different aspects of a patient’s medical history and hospital stay:

## 1. ADMISSIONS.csv
Contains admission-related details including timestamps, admission/discharge types and locations, insurance, and outcomes.
Key Columns: ADMITTIME, DISCHTIME, ADMISSION_TYPE, HOSPITAL_EXPIRE_FLAG

## 2. PATIENTS.csv
Provides demographic information such as age, gender, and mortality status.
Key Columns: GENDER, DOB, DOD, EXPIRE_FLAG

## 3. DIAGNOSES_ICD.csv
Stores ICD-9 diagnosis codes assigned during hospital stays.
Key Columns: HADM_ID, ICD9_CODE

## 4. PROCEDURES_ICD.csv
Lists ICD-9 procedure codes for treatments/procedures performed during admissions.
Key Columns: HADM_ID, ICD9_CODE

## 5. PRESCRIPTIONS.csv
Contains information on medications prescribed during hospitalizations.
Key Columns: DRUG, STARTDATE, ENDDATE, HADM_ID

## 6. NOTEEVENTS.csv
Captures unstructured clinical documentation including discharge summaries, physician notes, and nursing progress notes. This dataset is used for natural language processing (NLP) tasks like embedding generation with ClinicalBERT.
Key Columns: HADM_ID, CHARTDATE, CATEGORY, TEXT

# 3.2 Data Cleaning and Preprocessing

## Handling Missing Data

**DEATHTIME:** Missing for ~53,122 records because many patients did not die during their admission. This was expected and not imputed.

**LANGUAGE:** Missing for a large portion of patients. Imputed using a constant value "Unknown".

**MARITAL_STATUS: **Missing in 10,128 records. Filled with "Unknown" or categorized accordingly.

**RELIGION:** Also had missing values and was imputed similarly using a constant "Unknown".

**EDREGTIME and EDOUTTIME:** Missing in several records and considered irrelevant for the analysis (related to emergency department registration times). These columns were dropped entirely.

## Imputation Strategy

Constant Imputation: Used for categorical fields such as LANGUAGE, MARITAL_STATUS, and RELIGION. Missing values were replaced with a placeholder like "Unknown" to retain the records without introducing bias.

## Column Dropping

Columns with excessive or non-informative missing values (e.g., EDREGTIME, EDOUTTIME) were dropped to reduce noise and maintain model relevance.

## Overall Workflow

Identified missing values.

Determined relevance based on domain knowledge.

Applied appropriate imputation or dropped columns.

Ensured cleaned data was consistent, complete, and ready for modeling.

# 4 Visualizations

## 1: Correlation Heatmap
![image](https://github.com/user-attachments/assets/8c67f821-7143-496e-b618-9b5e381c1db3)

- Length of Stay (LOS_DAYS): LOS_DAYS demonstrates a moderate positive correlation with both NUM_PROCEDURES and NUM_DIAGNOSES, suggesting that patients undergoing more procedures or diagnosed with more conditions tend to have longer hospital stays.

- Mortality Outcome (HOSPITAL_EXPIRE_FLAG): This variable shows a weak negative correlation with LOS_DAYS (approximately -0.17), indicating that shorter hospital stays are slightly associated with higher mortality rates.

- Multicollinearity Assessment: Most feature correlations are below 0.6, suggesting minimal multicollinearity—a favorable condition for machine learning models, as it reduces the risk of misleading relationships and model instability.

![image](https://github.com/user-attachments/assets/c2d4e465-f9ef-4fdd-ade3-651da4fc2edd)

- Out of a total of 58,976 patient cases, 12,456 individuals—approximately 21%—were readmitted to the hospital.

- The remaining 46,520 patients (79%) did not experience a subsequent admission.

- This notable readmission rate underscores the need for proactive discharge planning and timely interventions.

- In effect, nearly one in five patients returned post-discharge, emphasizing the critical role of monitoring and mitigating risks at the point of discharge.

![image](https://github.com/user-attachments/assets/1c7114b3-c19f-4372-ac22-d843e60b67be)

- **Mechanical Ventilation (HAS_VENT):** Used in 55.1% of patients who died, mechanical ventilation shows a strong link to fatal outcomes, suggesting it is a key indicator of critical illness.

- **Sepsis (HAS_SEPSIS):** Involved in 23.6% of deaths, sepsis stands out as a major contributing factor, emphasizing the life-threatening nature of severe infections.

- **Diabetes (HAS_DIABETES):** Found in 21.3% of deceased cases, diabetes also plays a significant role in mortality risk.

- Overall, the data indicates that more than half of the patients who died required ventilation, and nearly a quarter had sepsis—underscoring how serious medical conditions heavily impact hospital mortality.

![image](https://github.com/user-attachments/assets/824ad01f-633d-417d-9a9f-31955c7843ca)

"Urgent and emergency admissions are associated with longer hospital stays and more diagnoses, highlighting their higher clinical complexity compared to elective cases."

![image](https://github.com/user-attachments/assets/ab5f0626-28c2-492a-a024-7f2b4b98d360)

publicly insured patients consistently have higher mortality rates across all procedure groups, suggesting possible disparities in outcomes related to insurance coverage and patient severity.

# 5 Feature Engineering

## 1. Continuous Features

LOS_DAYS: Length of stay in the hospital (in days).

NUM_DIAGNOSES: Number of ICD-9 diagnosis codes assigned.

NUM_PROCEDURES: Number of procedures performed during admission.

## 2. Binary Flags

HAS_SEPSIS: Indicates whether the patient was diagnosed with sepsis.

HAS_DIABETES: Indicates presence of diabetes in diagnosis codes.

HAS_VENT: Indicates if mechanical ventilation was used during stay.

## 3. Insurance Type (Derived from INSURANCE)

INSURANCE_PUBLIC: Includes government programs (e.g., Medicare, Medicaid).

INSURANCE_PRIVATE: Includes private or employer-sponsored plans.

INSURANCE_OTHER: Self-pay, no insurance, or other forms.

## 4. Discharge Disposition (Grouped from DISCHARGE_LOCATION)

DISCHARGE_GROUP_HOME: Patient discharged to home/self-care.

DISCHARGE_GROUP_DEATH: Discharge resulted in in-hospital death.

## 5. Categorical Features

ADMISSION_TYPE: Nature of admission (e.g., emergency, elective).

ADMISSION_LOCATION: Department/location of initial care.

ETHNICITY: Self-reported ethnicity of the patient.

## 6. Targets

HOSPITAL_EXPIRE_FLAG: Binary target; 0 = survived, 1 = died during hospital stay.

FUTURE_ADMISSION: Binary target; 1 = readmitted, 0 = not readmitted.

## 7. Preprocessing Pipeline

DateTime Conversion : ADMITTIME, DISCHTIME converted to datetime objects for time-based feature engineering (e.g., length of stay).

Standardization: Applied to continuous features (LOS_DAYS, NUM_DIAGNOSES, NUM_PROCEDURES).

One-hot encoding: Applied to categorical features (ADMISSION_TYPE, ADMISSION_LOCATION, ETHNICITY), with the first category dropped to avoid multicollinearity.

Scaling
Applied StandardScaler or MinMaxScaler to continuous features (LOS_DAYS, NUM_DIAGNOSES, NUM_PROCEDURES) to normalize the scale.

Passthrough
Applied to Binary features (HAS_SEPSIS, HAS_DIABETES, HAS_VENT, insurance and discharge flags) passed without transformation.

Automation
Entire preprocessing flow implementation is managed using ColumnTransformer and Pipeline from scikit-learn for modularity, consistency and efficient transformation.


# 6 Model Development

## 6.1. Objective

The main objectives are to:

- Predict in-hospital mortality using structured clinical features such as length of stay, number of diagnoses, number of procedures, and flags for conditions like sepsis and diabetes.

- Predict 30-day readmission for survivors using similar features.

- Compare and evaluate multiple models: Logistic Regression, XGBoost, CatBoost, and Decision Tree.

- Build an ensemble Voting Classifier using a weighted soft voting approach.

- Apply SHAP (Shapley Additive Explanations) to provide explainability and transparency of model decisions.

- Visualize and interpret performance using metrics like precision, recall, AUC, MCC, and confusion matrices.

## 6.2. Models Implemented

## 1. Logistic Regression
- Regularization: L1/L2 penalties applied to avoid overfitting.
- C: Regularization strength controlled by parameter C.

- Solver: liblinear used to fit the model.

- Class Weight: Set to 'balanced' to account for class imbalance.

- Evaluation Metrics: ROC AUC, Precision/Recall, MCC.

## 2. XGBoost Classifier
- Tuning Method: Hyperparameters tuned with RandomizedSearchCV.

- Grid Search with Key Hyperparameters:
-     - n_estimators, max_depth, learning_rate, subsample.

- Best AUC for Mortality: 0.872.

- Interpretability: SHAP used for model interpretation and feature importance.

## 3. CatBoost Classifier
- Categorical Feature Handling: CatBoost handles categorical variables natively without needing to encode them beforehand.

- Tuning Method: Hyperparameters tuned via grid search, focusing on:
-     - iterations, learning_rate, depth, l2_leaf_reg.

- Best AUC for Mortality: 0.876.

- Key Strength: Strong performance, especially for mortality prediction.

## 4. Decision Tree Classifier
- Grid Search: Hyperparameters like max_depth, min_samples_split, min_samples_leaf, and criterion tuned.

- Evaluation: Interpretable model.

- Best AUC for Mortality: 0.848.

## 5. Voting Classifier (Ensemble Model)

- Combination of Models:
 - Logistic Regression (LR), XGBoost (XGB), and Decision Tree (DT).

- Voting Method: 'Soft' voting, with weights (LR=1, XGB=2, DT=1).

- Best AUC for Mortality: 0.862.

## 6.3. Evaluation Metrics

For both mortality and readmission predictions, the following evaluation metrics were used:

ROC, AUC: To evaluate model performance.

- Precision and Recall: Important for understanding the model’s ability to handle imbalanced data.

- Matthews Correlation Coefficient (MCC): Measures the quality of binary classifications.

- Confusion Matrix: For visualizing the classification results.

- SHAP Explanation Plots: To explain model predictions and feature importance.

## 6.4. Visualizations
- ROC and PR Curves: To visualize trade-offs between precision and recall.

- Confusion Matrices: Displayed using ConfusionMatrixDisplay to evaluate the classification performance.

- SHAP Plots:
-     - SHAP summary and bar plots to show feature importance and contribution.

## 6.5. Model Interpretability Using SHAP
- SHAP for Interpretability:

-     - Used shap.Explainer for general models and TreeExplainer for tree-based models like XGBoost, CatBoost, and Decision Trees for SHAP values.

      - SHAP Plots: Summary plots and bar plots were used to highlight the most important features affecting model decisions i.e Feature Ranking.

      - Ensured SHAP inputs matched post-transformation feature arrays for accurate explanation.

## 6.6. Preprocessing and Deployment Pipeline

- Preprocessing: Data preprocessing (e.g., handling missing values, encoding categorical features) was wrapped in a Pipeline for streamlined execution.

- Deployment: The model pipeline is deployment-ready, suitable for integration into a Streamlit or Flask application.

- Future plans include real-time prediction with ClinicalBERT note embeddings

-     - The pipeline ensures consistency in preprocessing during both training and inference.

## 6.7. Known Issues

- Readmission Prediction: Class imbalance is a challenge for readmission prediction, with low sensitivity. This needs attention, potentially through threshold tuning or SMOTE (Synthetic Minority Over-sampling Technique) to balance the classes.

- Ensemble Interpretability: While SHAP is used for individual components, direct interpretability for the ensemble classifier is not straightforward.

## 6.8. Technologies Used
- Python: For scripting and model development.

- Libraries:
-     - pandas, numpy for data handling.
-     - matplotlib, seaborn for visualizations.
-     - SHAP for model interpretability.
-     - xgboost, catboost, scikit-learn for machine learning models.

## 6.9. Future Work
- Threshold Optimization: To improve the recall for readmission prediction, threshold optimization is planned.

- Temporal Validation: Simulate real-world prediction scenarios by incorporating temporal validation.

- Clinical Note Embeddings: Integrating ClinicalBERT embeddings for better prediction signals from clinical notes.

- Model Deployment: Plan to deploy the model using Flask or FastAPI for real-time scoring.

- AutoML and Stacking: Future experiments will involve stacking classifiers or using AutoML tools to further optimize model performance.

## 6.10. Installation Requirements
To run the code and models, ensure the following libraries are installed:

- pip install pandas numpy matplotlib seaborn shap xgboost catboost scikit-learn

## 6.11. Features Used

Features Used
| Feature Type | Features                                                               |
| ------------ | ---------------------------------------------------------------------- |
| Numerical    | Length of stay, number of diagnoses, number of procedures              |
| Binary       | Sepsis flag, diabetes, ventilator use, insurance type, discharge group |
| Categorical  | Admission type, admission location, ethnicity                          |

## 6.12. Model Results Summary

Model Results Summary

| Model                 | AUC      | MCC      | Sensitivity | Specificity |
| --------------------- | -------- | -------- | ----------- | ----------- |
| Logistic              | 0.83     | 0.35     | 0.73        | 0.79        |
| XGBoost               | 0.87     | 0.41     | 0.26        | 0.99        |
| CatBoost              | 0.88     | 0.42     | 0.28        | 0.99        |
| Decision Tree         | 0.85     | 0.40     | 0.79        | 0.77        |
| **Voting Classifier** | **0.86** | **0.40** | 0.30        | 0.98        |

Readmission Prediction Results (Survivors Only)

| Model                 | AUC      | MCC      | Sensitivity | Specificity |
| --------------------- | -------- | -------- | ----------- | ----------- |
| Logistic              | 0.70     | 0.24     | 0.72        | 0.56        |
| XGBoost               | 0.71     | 0.12     | 0.05        | 0.99        |
| CatBoost              | 0.72     | 0.14     | 0.08        | 0.98        |
| Decision Tree         | 0.69     | 0.24     | 0.72        | 0.55        |
| **Voting Classifier** | **0.69** | **0.17** | 0.20        | 0.92        |

## 6.13. Best Model

![image](https://github.com/user-attachments/assets/4cfead5a-c2bb-4280-8b87-27e8d88aebc6)

## 6.13.1 Best Mortality Prediction Model: CatBoostClassifier
### Performance Highlights:

- AUC: 0.876 – Highest among all models, indicating strong discrimination between mortality and survival.

- MCC: 0.421 – Best balance of true and false positives/negatives in an imbalanced dataset.

- Sensitivity: 0.278 – Slightly better than XGBoost (0.26), capturing more actual deaths.

- Specificity: 0.99 – Very few false positives.

### Why CatBoost?

- Natively handles categorical features (e.g., admission type, ethnicity) without preprocessing.

- Robust performance even on skewed clinical datasets.

- Outperformed both XGBoost and ensemble models in key metrics critical to mortality prediction.

### Interpretability:

- SHAP values clearly identified impactful features such as:

-- Number of diagnoses

-- Ventilator use

-- Sepsis flag

-- Insurance type

- Used shap.TreeExplainer and summary plots for feature importance and decision rationale.

## 6.13.2 Best Readmission Prediction Model: Logistic Regression

- Performance Highlights:

-- MCC: 0.236 – Highest among all models, indicating best balance despite class imbalance.

-- Sensitivity: 0.719 – Captured the most actual readmissions, crucial for reducing re-hospitalization.

-- AUC: 0.697 – Reasonable discrimination capability.

- Why Logistic Regression?

-- Outperformed complex models like XGBoost and CatBoost in terms of recall for readmission.

-- Simpler and interpretable, making it suitable for clinical deployment and transparency.

-- Balanced trade-off between recall and false positives, prioritizing patient safety.

- Interpretability:
  
-- Coefficients and SHAP values revealed:

--- High number of diagnoses and sepsis flag increased readmission risk.

--- Insurance type and length of stay also contributed significantly.

## Summary: Model Selection Rationale

| Task            | Best Model              | Justification                                                    |
| --------------- | ----------------------- | ---------------------------------------------------------------- |
| **Mortality**   | **CatBoostClassifier**  | Highest AUC & MCC; strong performance on minority class (deaths) |
| **Readmission** | **Logistic Regression** | Best sensitivity; maximized recall for critical patient outcome  |

### Metric Prioritization (for Healthcare):

- ROC AUC: Measures ability to distinguish positive vs negative classes.

- MCC (Matthews Correlation Coefficient): Ideal for imbalanced data; considers all error types.

- Sensitivity (Recall): Crucial for catching true critical cases (e.g., deaths or readmissions).



# 7. Clinical Note Embeddings

## Note Types Selected

- Categories: Focused on Discharge Summary, Physician Notes, and Nursing Notes

- Rationale: These are clinically rich and relevant for patient outcome prediction.

## Text Preprocessing

- Converted text to lowercase

- Removed:

      Punctuation

      Special characters

      Extra whitespace

- Goal: Reduce noise and ensure consistency

## Handling Note Lengths

- ClinicalBERT has a 512-token limit

- Solution: Truncated notes to the first 512 tokens for model compatibility

## Efficient Data Loading

- Used chunked reading (chunksize=10,000) to handle large datasets efficiently and avoid memory issues

## ClinicalBERT Embedding Generation

### Model Selection: ClinicalBERT

- Domain-specific BERT trained on MIMIC-III clinical notes

- Superior to general BERT models for clinical text understanding

### Model Initialization

- Used HuggingFace’s AutoTokenizer and AutoModel

- Handled tokenization and model architecture automatically

### Embedding Extraction

- Passed each cleaned note through ClinicalBERT

- Extracted final hidden states

- Applied mean pooling across all token embeddings (excluding [CLS], [SEP], etc.)

- Result: One 768-dimensional vector per note

### Merging with Structured Data

- Embedding vectors stored in embedding_df with corresponding HADM_ID

- Ready for merging with structured features (e.g., vitals, labs) for model input

## Clinical Note Embedding Generation using ClinicalBERT

### Dimensionality Reduction & Evaluation of Clinical Note Embeddings

### t-SNE Projection

![image](https://github.com/user-attachments/assets/d5c5b786-d64b-457f-b062-49a960e00125)

### Parameters:

-- Perplexity: 10

-- Iterations: 1500

### Metrics:

-- Silhouette Score: 0.060

-- Davies-Bouldin Index: ~4.00

### Observations:

-- High cluster overlap

-- Poor separation of Nursing, Physician, and Discharge Summary

-- Clinical notes share overlapping medical vocabulary, leading to semantic blending

## UMAP Projection

![image](https://github.com/user-attachments/assets/a20504e4-391d-4559-94cc-dec6d5395b3a)

### Parameters:

-- n_neighbors: 10

-- min_dist: 0.1

### Metrics:

-- Silhouette Score: 0.060

-- Davies-Bouldin Index: ~4.00

### Observations:

-- Better local structure preservation

-- Smaller, tighter clusters

-- Captured semantic proximity more effectively than t-SNE

## Embedding Impact on Model Performance

### Embedding Input Flow:

Clinical Note ➝ ClinicalBERT ➝ 768-dim Vector ➝ Concatenate with Structured Features ➝ Prediction

### Features Combined:

- Structured: LOS, lab flags, age, etc.

- Unstructured: Clinical notes via ClinicalBERT

### Performance Gains:

- Improved Sensitivity and MCC

- Models became clinically more useful, especially in identifying critical outcomes

## Evaluation Metrics – Before vs. After Embeddings

![image](https://github.com/user-attachments/assets/48a31583-a419-4bb5-964f-6ff312d897c4)

## Sample Results

![image](https://github.com/user-attachments/assets/9bfb0883-8a45-4f4e-8f1d-e5bc246b25ee)

# Inference

INPUT:
Structured values: LOS, lab flags, age, etc.
Unstructured note (raw clinical text)

![image](https://github.com/user-attachments/assets/3d472a04-0eb8-40b2-b32a-0f69e742b746)

### Output:

- Mortality risk

- Readmission likelihood

# 8 Conclusion

This project successfully demonstrates the potential of combining structured and unstructured electronic health record (EHR) data from the MIMIC-III database to predict critical clinical outcomes such as in-hospital mortality and 30-day readmission. By integrating features from diagnosis codes, procedures, prescriptions, demographics, and admission data with ClinicalBERT embeddings from clinical notes, we built an end-to-end machine learning pipeline using XGBoost classifiers. The fusion of deep contextual embeddings with traditional clinical variables led to noticeable improvements in model performance, particularly in terms of AUC and Matthews Correlation Coefficient (MCC). This highlights the importance of leveraging unstructured clinical text for better patient risk stratification.

# 8.1 Future Scope

1. Temporal Modeling:
Extend the model to incorporate time-series analysis using sequential models (e.g., LSTMs or Transformers) to capture progression over a patient’s stay.

2. Model Generalization:
Test the pipeline on external EHR datasets or MIMIC-IV to evaluate robustness and generalizability beyond the MIMIC-III cohort.

3. Multi-Modal Fusion Models:
Explore the integration of lab results, imaging reports, and vitals data with notes and structured features using multi-modal deep learning architectures.

4. Intelligent Clinical Chatbot:
Develop a clinical assistant chatbot fine-tuned on patient notes and structured data to answer queries such as:

"What is the most recent diagnosis?"

"Was the patient readmitted?"

"Summarize the patient’s current medications."
This chatbot can assist healthcare professionals by automatically extracting insights from patient histories, improving workflow efficiency and reducing manual EHR search time.

# 9 References 

MIMIC-III Clinical Database v1.4:
https://physionet.org/content/mimiciii/

Research Papers
CPLLM: Clinical Prediction with Large Language Models:
https://arxiv.org/abs/2309.11295

A Multimodal Transformer: Fusing Clinical Notes with Structured EHR Data for Interpretable In-Hospital Mortality Prediction:
https://arxiv.org/abs/2208.10240

ClinicalBERT & NLP Models
ClinicalBERT Model Card:
https://huggingface.co/medicalai/ClinicalBERT

Bio_ClinicalBERT Model Card:
https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT

ClinicalBERT GitHub Repository:
https://github.com/kexinhuang12345/clinicalBERT

ClinicalBERT Research Paper:
https://arxiv.org/abs/1904.05342

Streamlit
Streamlit Official Documentation:
https://docs.streamlit.io/

Streamlit GitHub Repository:
https://github.com/streamlit/streamlit

Streamlit Official Website:
https://streamlit.io/

Hugging Face Transformers
Transformers Library Documentation:
https://huggingface.co/docs/transformers/en/index

Transformers GitHub Repository:
https://github.com/huggingface/transformers

Transformers Research Paper:
https://arxiv.org/abs/1910.03771

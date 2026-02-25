from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent

MORTALITY_MODEL_PATH = BASE_DIR / "mortality_model.pkl"
READMISSION_MODEL_PATH = BASE_DIR / "readmission_catboost_model.pkl"

USE_BIOCLINICAL_BERT = os.getenv("USE_BIOCLINICAL_BERT", "false").lower() == "true"
BERT_MODEL_NAME = os.getenv("BERT_MODEL_NAME", "emilyalsentzer/Bio_ClinicalBERT")
APP_ENV = os.getenv("APP_ENV", "production")

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, UTC
import hashlib
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from config import MORTALITY_MODEL_PATH, READMISSION_MODEL_PATH, USE_BIOCLINICAL_BERT, BERT_MODEL_NAME



@dataclass
class PredictionMeta:
    created_at: str
    embedding_mode: str


class RiskPredictor:
    def __init__(self) -> None:
        self.mortality_model = joblib.load(MORTALITY_MODEL_PATH)
        self.readmission_model = joblib.load(READMISSION_MODEL_PATH)
        self.embedding_mode = "hash"
        self._tokenizer = None
        self._bert_model = None

        if USE_BIOCLINICAL_BERT:
            self._try_load_bert()

    def _try_load_bert(self) -> None:
        try:
            from transformers import AutoTokenizer, AutoModel  # type: ignore
            import torch  # type: ignore

            self._torch = torch
            self._tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
            self._bert_model = AutoModel.from_pretrained(BERT_MODEL_NAME)
            self.embedding_mode = "clinicalbert"
        except Exception:
            self.embedding_mode = "hash"

    def _hash_embedding(self, text: str) -> np.ndarray:
        # Deterministic 768-length embedding fallback when transformers are unavailable.
        vals = []
        seed = text.encode("utf-8", errors="ignore")
        digest = hashlib.sha256(seed).digest()
        while len(vals) < 768:
            digest = hashlib.sha256(digest + seed).digest()
            for b in digest:
                vals.append((b / 255.0) * 2 - 1)
                if len(vals) == 768:
                    break
        return np.array(vals, dtype=np.float32)

    def _generate_embedding(self, text: str) -> np.ndarray:
        if self.embedding_mode == "clinicalbert" and self._tokenizer is not None and self._bert_model is not None:
            tokens = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            )
            with self._torch.no_grad():
                output = self._bert_model(**tokens)
            return output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

        return self._hash_embedding(text)

    def _risk_tier(self, mort_prob: float, readm_prob: float) -> Tuple[str, str]:
        if mort_prob < 0.25:
            mort_risk = "Low"
        elif mort_prob < 0.60:
            mort_risk = "Moderate"
        else:
            mort_risk = "High"

        if readm_prob < 0.30:
            readm_risk = "Low"
        elif readm_prob < 0.60:
            readm_risk = "Moderate"
        else:
            readm_risk = "High"

        return mort_risk, readm_risk

    def predict(self, payload: Dict[str, Any]) -> Tuple[Dict[str, Any], PredictionMeta]:
        input_dict = {
            "LOS_DAYS": payload["los_days"],
            "NUM_DIAGNOSES": payload["num_diagnoses"],
            "NUM_PROCEDURES": payload["num_procedures"],
            "HAS_SEPSIS": int(payload["has_sepsis"]),
            "HAS_DIABETES": int(payload["has_diabetes"]),
            "HAS_VENT": int(payload["has_vent"]),
            "INSURANCE_PRIVATE": int(payload["insurance"] == "PRIVATE"),
            "INSURANCE_PUBLIC": int(payload["insurance"] == "PUBLIC"),
            "INSURANCE_OTHER": int(payload["insurance"] == "OTHER"),
            "DISCHARGE_GROUP_OTHEROTHER": int(payload["discharge_group"] == "OTHER"),
            "DISCHARGE_GROUP_HOME": int(payload["discharge_group"] == "HOME"),
            "DISCHARGE_GROUP_DEATH": int(payload["discharge_group"] == "DEATH"),
            "DISCHARGE_GROUP_FACILITY": int(payload["discharge_group"] == "FACILITY"),
            "DISCHARGE_GROUP_OTHER": int(payload["discharge_group"] == "OTHER"),
            "ADMISSION_TYPE": payload["admission_type"],
            "ADMISSION_LOCATION": payload["admission_location"],
            "ETHNICITY": payload["ethnicity"],
        }

        df = pd.DataFrame([input_dict])
        embedding = self._generate_embedding(payload["clinical_note"])
        embedding_df = pd.DataFrame(embedding.reshape(1, -1), columns=[str(i) for i in range(768)])
        features = pd.concat([df.reset_index(drop=True), embedding_df], axis=1)

        mort_prob = float(self.mortality_model.predict_proba(features)[0][1])
        readm_prob = float(self.readmission_model.predict_proba(features)[0][1])

        mort_risk, readm_risk = self._risk_tier(mort_prob, readm_prob)
        result = {
            "mortality_probability": mort_prob,
            "mortality_risk_tier": mort_risk,
            "readmission_probability": readm_prob,
            "readmission_risk_tier": readm_risk,
            "embedding_mode": self.embedding_mode,
        }
        meta = PredictionMeta(
            created_at=datetime.now(UTC).isoformat(),
            embedding_mode=self.embedding_mode,
        )

        return result, meta

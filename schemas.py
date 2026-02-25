from pydantic import BaseModel, Field
from typing import Literal

InsuranceType = Literal["PRIVATE", "PUBLIC", "OTHER"]
DischargeGroup = Literal["HOME", "FACILITY", "DEATH", "OTHER"]
AdmissionType = Literal["EMERGENCY", "URGENT", "ELECTIVE"]
AdmissionLocation = Literal["EMERGENCY ROOM ADMIT", "TRANSFER", "CLINIC REFERRAL"]
Ethnicity = Literal["WHITE", "BLACK", "HISPANIC", "ASIAN", "OTHER"]


class PredictRequest(BaseModel):
    los_days: int = Field(ge=0, le=100)
    num_diagnoses: int = Field(ge=0, le=50)
    num_procedures: int = Field(ge=0, le=50)
    has_sepsis: bool
    has_diabetes: bool
    has_vent: bool
    insurance: InsuranceType
    discharge_group: DischargeGroup
    admission_type: AdmissionType
    admission_location: AdmissionLocation
    ethnicity: Ethnicity
    clinical_note: str = Field(min_length=1, max_length=10000)


class RiskPrediction(BaseModel):
    mortality_probability: float
    mortality_risk_tier: str
    readmission_probability: float
    readmission_risk_tier: str
    embedding_mode: str

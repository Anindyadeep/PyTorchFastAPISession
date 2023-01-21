from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class InferenceInput(BaseModel):
    """
    Input values for model inference
    """
    age : int = Field(..., example=30, title='Age of the person')
    sex  : str =  Field(..., example='male', title='The sex of the person')
    chest_pain_type : int = Field(..., example=1, title='Type of chest pain values must be 0/1/2')
    resting_blood_pressure : float = Field(..., example=154.0, title='The value of resting blood pressure from report')
    cholesterol : bool= Field(..., example=True, title='Whether person have cholesterol or not')
    fasting_blood_sugar : bool = Field(..., example=False, title='Whether person have blood sugar or not')
    resting_electro_cardio_graphic_result: int = Field(..., example=1, title='Values must be 0/1/2')
    max_heart_rate_achieved : float = Field(..., example=175, title='Max heart rate reached by the patient')
    exercise_induced_angina: bool = Field(..., example=True, title='Exercise induced angina')
    old_peak : float = Field(..., example=1.6, title='Old peak of the patient')


class InferenceResult(BaseModel):
    """
    Inference result from the model
    """
    prediction : str = Field(..., example='Person has heart attack', title='Written prediction of occurrence of heart attack') 
    confidence : float  = Field(..., example=0.99, title='Confidence of the model prediction')


class InferenceResponse(BaseModel):
    """
    Output response for model inference
    """
    error: bool = Field(..., example=False, title='Whether there is error')
    prediction: str = Field(..., example='Person has heart attack', title='Model prediction')
    confidence : float  = Field(..., example=0.99, title='Confidence of the model prediction')

class ErrorResponse(BaseModel):
    """
    Error response for the API
    """
    error: bool = Field(..., example=True, title='Whether there is error')
    message: str = Field(..., example='', title='Error message')
    traceback: str = Field(None, example='', title='Detailed traceback of the error')

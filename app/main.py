import os 
import sys 
import json
import uvicorn
from joblib import load
from fastapi.logger import logger
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError

import torch
from config import CONFIG
from inference import Inference
from Models.baseline_model import Model
from schemas import InferenceInput, InferenceResponse, ErrorResponse
from exceptions import validation_exception_handler, python_exception_handler

app = FastAPI(title="Sample ML App using FastAPI", version="0.0.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"])
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, python_exception_handler)


@app.on_event('startup')
async def startup_event():
    """
    All the initialization of variables and models are to be done here
    """
    logger.info("=> Running environment: {}".format(CONFIG["ENV"]))
    logger.info("=> PyTorch using device: {}".format(CONFIG["DEVICE"]))

    
    model = Model(in_features=CONFIG['IN_FEATURES'])
    model.load_state_dict(
        torch.load(CONFIG['MODEL_PATH'], map_location=CONFIG['DEVICE'])
    )

    model.eval() 
    logger.info("=> Model loaded successfully")
    app.package = {
        'model' : model, 
        'scaler': load(CONFIG['SCALAR_PATH'])
    }

    app.inference = Inference(app.package)
    logger.info("=> Server listening on PORT")


@app.get('/')
def root_dir():
    return {
        'Hello world': 'Welcome to FastAPI Tutorial to make ML powered REST APIs'
    }


@app.post('/api/v1/predict', response_model=InferenceResponse, responses={422: {'model': ErrorResponse}, 500: {'model': ErrorResponse}})
def predict(request: Request, body: InferenceInput):
    logger.info(f'input: {body}')

    request_params = {
            'age' : body.age, 
            'sex' : body.sex, 
            'chest_pain_type': body.chest_pain_type, 
            'resting_blood_pressure': body.resting_blood_pressure, 
            'cholesterol': body.cholesterol, 
            'fasting_blood_sugar': body.fasting_blood_sugar,
            'resting_electro_cardio_graphic_result': body.resting_electro_cardio_graphic_result, # between 0-2 
            'max_heart_rate_achieved': body.max_heart_rate_achieved, 
            'exercise_induced_angina': body.exercise_induced_angina, 
            'old_peak': body.old_peak, 
    }

    logger.info(request_params)

    response = app.inference.predict(request_params)
    logger.info(f'response: {response}')
    return {
        'error': False, 
        'prediction': response['prediction'], 
        'confidence': response['confidence'],
    }

# GET Method to Information About the API 

@app.get('/about')
def show_about():
    """
    Get deployment information, for debugging
    """

    return {
        "sys.version": sys.version,
        "torch.__version__": torch.__version__,
        "torch.cuda.is_available()": torch.cuda.is_available(),
        "USING CUDA": CONFIG['USE_CUDE_IF_AVAILABLE']
    }


if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
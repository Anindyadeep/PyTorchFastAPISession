import torch 
import numpy as np 
from config import CONFIG

from typing import Dict, Any

class Inference:
    def __init__(self, package: dict) -> None:
        self._categorical_mapping = {
            'chest_pain_type' : {
                0 : [1, 0, 0], 
                1 : [0, 1, 0], 
                2 : [0, 0, 1]
            }, 

            'resting_electro_cardio_graphic_result': {
                0 : [1, 0], 
                1 : [0, 1], 
                2 : [1, 1]
            }
        }
        self._scaler, self._model = package['scaler'], package['model'].to(CONFIG['DEVICE'])
        self._class_mapping = {
            0 : 'less chance of heart attack', 
            1 : 'more chance of heart attack'
        }
    
    def _flatten_values(self, values):
        value_list = []
        for val in values:
            if type(val) == list:
                value_list += val
            else:
                value_list.append(val)
        return value_list

    def _preprocess_request(self, request : Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess request

        Args:
            package (dict): The package used by FastAPI to get the scaler 
            request (Dict[str, Any]): The request we will be getting 

        Returns:
            Dict[str, Any]: Preprocessed request
        """
        

    
        request['sex'] = 0 if request['sex'] == 'male' else 1
        request['cholesterol'] = int(request['cholesterol'])
        request['fasting_blood_sugar'] = int(request['fasting_blood_sugar'])
        request['exercise_induced_angina'] = int(request['exercise_induced_angina'])

        request['chest_pain_type'] = self._categorical_mapping['chest_pain_type'][request['chest_pain_type']]
        request['resting_electro_cardio_graphic_result'] = self._categorical_mapping['resting_electro_cardio_graphic_result'][request['resting_electro_cardio_graphic_result']]

        request_values = self._flatten_values(request.values())
        features = self._scaler.fit_transform(np.array(request_values).reshape(-1, 1)).T
        return torch.tensor(features, dtype=torch.float32).to(CONFIG['DEVICE'])
    

    def predict(self, request: dict) -> dict:
        preprocessed_request = self._preprocess_request(request)
        predictions = self._model(preprocessed_request)
        predicted_class, class_probability = torch.argmax(predictions).item(), torch.max(predictions).item()
        return {
            'prediction' : self._class_mapping[predicted_class],
            'confidence' : class_probability
        }


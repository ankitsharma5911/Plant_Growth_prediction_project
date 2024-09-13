import sys,os
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass


    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        

class CustomData:
    def __init__(self,
                 Soil_Type:str,
                 Sunlight_Hours:float,
                 Water_Frequency:str,
                 Fertilizer_Type:str,
                 Temperature:float,
                 Humidity:float):
        self.soil_type = Soil_Type,
        self.sunlight_hours = Sunlight_Hours
        self.water_frequency = Water_Frequency
        self.fertilizer_type = Fertilizer_Type
        self.temperature = Temperature
        self.humidity = Humidity

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'soil_type':[self.soil_type],
                'sunlight_hours':[self.sunlight_hours]
            }
        except Exception as e:
            CustomException(e,sys)




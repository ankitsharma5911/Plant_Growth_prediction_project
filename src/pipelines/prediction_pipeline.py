import sys,os,pandas as pd
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
        self.soil_type = Soil_Type
        self.sunlight_hours = Sunlight_Hours
        self.water_frequency = Water_Frequency
        self.fertilizer_type = Fertilizer_Type
        self.temperature = Temperature
        self.humidity = Humidity

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Soil_Type':[self.soil_type],
                'Sunlight_Hours':[self.sunlight_hours],
                'Water_Frequency':[self.water_frequency],
                'Fertilizer_Type':[self.fertilizer_type],
                'Temperature':[self.temperature],
                'Humidity':[self.humidity]
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            logging.info(df.head(5))
            return df
        except Exception as e:
            CustomException(e,sys)

if __name__=="__main__":
    data=CustomData(
            Soil_Type='loam',
            Sunlight_Hours = 5,
            Water_Frequency = 'weekly',
            Fertilizer_Type = 'none',
            Temperature = 20,
            Humidity = 50
        )
    final_new_data=data.get_data_as_dataframe()
    predict_pipeline=PredictPipeline()
    
    pred=predict_pipeline.predict(final_new_data)

    print(pred) 



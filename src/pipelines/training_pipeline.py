import sys
import os
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion 
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer


if __name__ == "__main__":
    obj = DataIngestion()

    train_data_path,test_data_path = obj.initialize_data_ingestion()    

    print(train_data_path,test_data_path)

    data_transformation = DataTransformation()
    
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data_path,test_data_path)

    model_traner = ModelTrainer()
    model_traner.initate_model_training(train_arr,test_arr)





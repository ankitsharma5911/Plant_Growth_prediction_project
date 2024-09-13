import sys,os,  numpy as np,  pandas as pd, pickle
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from dataclasses import dataclass

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

## Data Transformation config
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')


## 
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    def get_data_transformation_object(self):

        try:
            logging.info('Data Transformation initiated')

            numerical_cols = ["Sunlight_Hours","Temperature","Humidity"]
            categorical_cols = ["Soil_Type","Water_Frequency","Fertilizer_Type"]

            ## Numerical Pipeline
            num_pipeline = Pipeline(
                        steps = [('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())
                ] )
            # Categorigal Pipeline
            cat_pipeline = Pipeline(
                steps = [('imputer',SimpleImputer(strategy='most_frequent')),('NominalEncoding',OneHotEncoder())
                ])


            preprocessor = ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_cols),
                ('cat_pipeline',cat_pipeline,categorical_cols)
            ])

            return preprocessor

            logging.info("pipeline completed")

        except Exception as e:
            raise CustomException (e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            
            # reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')
            
            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = "Growth_Milestone"

            # features into independent and dependent features

            input_feature_train_df = train_df.iloc[:,:-1]
            target_feature_train_df=train_df[target_column_name]
            
            
            input_feature_test_df = test_df.iloc[:,:-1]
            target_feature_test_df=test_df[target_column_name]
           
            # Apply transformation 

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)


            logging.info("Applying preprocessing object on training and testing datasets.")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            print(self.data_transformation_config.preprocessor_obj_file_path,)


            file_path=self.data_transformation_config.preprocessor_obj_file_path
            
            obj=preprocessing_obj

            save_object(file_path=file_path,obj=obj)



            logging.info('Processsor pickle in created and saved')

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )



        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)


# if __name__ == "__main__":
#     obj = DataTransformation()
#     obj.get_data_transformation_object()


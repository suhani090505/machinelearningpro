import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifact','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation=DataTransformationConfig()

    def get_data_transformer_obj(self):
        '''
        This function is responsible for creating the data transformer object,
        which includes pipelines for numerical and categorical features.
        '''
        try:
            numerical_columns=['writing_score','reading_score']
            categorical_columns=['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']

            # Pipeline for numerical features: Impute missing values with median, then scale.
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ("scalar",StandardScaler(with_mean=False))
                ]
            )

            # Pipeline for categorical features: Impute missing values with most frequent,
            # then apply OneHotEncoder, then scale.
            # handle_unknown='ignore' is added to OneHotEncoder to gracefully handle
            # categories not seen during training by encoding them as all zeros.
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("OneHotEncoder",OneHotEncoder(handle_unknown='ignore')), # Added handle_unknown='ignore'
                    ('scalar',StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Numerical features:{numerical_columns}")
            logging.info(f"Categorical features:{categorical_columns}")

            # Create a ColumnTransformer to apply different transformations to different columns.
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_tansformation(self,train_path,test_path):
        '''
        This function initiates the data transformation process by reading train/test data,
        applying preprocessing, and saving the preprocessor object.
        '''
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing obj")

            preprocessing_obj=self.get_data_transformer_obj()

            target_column="math_score"
            
            # Define categorical columns to ensure consistent lowercasing
            categorical_columns=['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']

            # Convert categorical columns to lowercase in both train and test dataframes
            # This ensures consistency with the input data in predict_pipeline.py
            for col in categorical_columns:
                if col in train_df.columns:
                    train_df[col] = train_df[col].astype(str).str.lower()
                if col in test_df.columns:
                    test_df[col] = test_df[col].astype(str).str.lower()

            input_feature_train_df=train_df.drop(columns=[target_column],axis=1)
            target_feature_train_df=train_df[target_column]

            input_feature_test_df=test_df.drop(columns=[target_column],axis=1)
            target_feature_test_df=test_df[target_column]


            logging.info(
                "Applying preprocessing obj on training dataframe and test dataframe"
            )

            # Fit the preprocessor on the training input features and transform them
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            # Transform the test input features using the fitted preprocessor
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            # Concatenate the processed features with the target column
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            # Save the fitted preprocessor object
            save_object(
                file_path=self.data_transformation.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)

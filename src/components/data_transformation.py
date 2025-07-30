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
    try:
      numerical_columns=['writing_score','reading_score']
      categorical_columns=['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']

      num_pipeline=Pipeline(
        steps=[
          ("imputer",SimpleImputer(strategy='median')),
          ("scalar",StandardScaler(with_mean=False))
        ]
      )
      cat_pipeline=Pipeline(
        steps=[
          ("imputer",SimpleImputer(strategy="most_frequent")),
          ("OneHotEncoder",OneHotEncoder()),
          ('scalar',StandardScaler(with_mean=False))
        ]
      )
      logging.info(f"Numerical features:{numerical_columns}")

      logging.info(f"Categorical features:{categorical_columns}")

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

    try:
      train_df=pd.read_csv(train_path)
      test_df=pd.read_csv(test_path)

      logging.info("read train and test data completed")

      logging.info("Obtaining preprocessing obj")

      preprocessing_obj=self.get_data_transformer_obj()

      target_column="math_score"
      numerical_columns=["writing_score","reading_score"]

      input_feature_train=train_df.drop(columns=[target_column],axis=1)
      target_feature_train=train_df[target_column]

      input_feature_test=test_df.drop(columns=[target_column],axis=1)
      target_feature_test=test_df[target_column]


      logging.info(
        "Applying preprocessing obj on training dataframe and test dataframe"
      )

      input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train)
      input_feature_test_arr=preprocessing_obj.transform(input_feature_test)

      train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train)]
      test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test)]


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

           
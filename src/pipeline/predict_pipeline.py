import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
  def __init__(self):
    pass
  def predict(self,features):
    try:
      model_path=os.path.join('artifact\model.pkl')
      preprocessor_path=os.path.join('artifact\preprocessor.pkl')
      model=load_object(file_path=model_path)
      preprocessor=load_object(file_path=preprocessor_path)
      data_scaled=preprocessor.transform(features)
      print("FEATURES TO TRANSFORM:", features)
      preds=model.predict(data_scaled)
      return preds
    except Exception as e:
      raise CustomException(e,sys)

class CustomData:
  def __init__(self, gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course, reading_score, writing_score):
        self.gender = gender.lower()
        self.race_ethnicity = race_ethnicity.lower()
        self.parental_level_of_education = parental_level_of_education.lower()
        self.lunch = lunch.lower()
        self.test_preparation_course = test_preparation_course.lower()
        self.reading_score = reading_score
        self.writing_score = writing_score



  def get_data_as_data_frame(self):
    try:
      custom_data_input_dict={
        "gender":[self.gender],
        "race_ethnicity":[self.race_ethnicity],
        "parental_level_of_education":[self.parental_level_of_education],
        "lunch":[self.lunch],
        "test_preparation_course":[self.test_preparation_course],
        "reading_score":[self.reading_score],
        "writing_score":[self.writing_score]
      } 
      return pd.DataFrame(custom_data_input_dict)
    
    except Exception as e:
      raise CustomException(e,sys)
    
      
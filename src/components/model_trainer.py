import sys
import os
from dataclasses import dataclass

from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression,Ridge,Lasso,LassoCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models

@dataclass
class ModelTrainerConfig:
  trained_model_file_path=os.path.join("artifact","model.pkl")

class ModelTrainer:
  def __init__(self):
    self.model_trainer_config=ModelTrainerConfig()

  def initiate_model_trainer(self,train_array,test_array):
    try:
      logging.info("Splitting training and test i/p data")
      X_train,Y_train,X_test,Y_test=(train_array[:,:-1],
                                     train_array[:,-1],
                                     test_array[:,:-1],
                                     test_array[:,-1])
      models={  
      "Linear Regression":LinearRegression(),
      "Lasso":Lasso(),
      "Ridge":Ridge(),
      "K Neighbors Regression":KNeighborsRegressor(),
      "Decision Tree":DecisionTreeRegressor(),
      "Random Forest Regression":RandomForestRegressor(),
      "XGBRegressor":XGBRegressor(),
      "CatBoost Regression":CatBoostRegressor(verbose=False),
      "AdaBoost":AdaBoostRegressor(),
      "GraidentBoost":GradientBoostingRegressor()
      }

      model_report:dict=evaluate_models(X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test,models=models)

      ##To get best model score from dict
      best_model_score=max(sorted(model_report.values()))

      ##To get best model name from dict
      best_model_name=list(model_report.keys())[
        list(model_report.values()).index(best_model_score)
      ]
      best_model=models[best_model_name]
      
      if best_model_score<0.6:
        raise CustomException("No best model found")
      logging.info("Best model found on both training and test dataset")

      save_object(
        file_path=self.model_trainer_config.trained_model_file_path,
        obj=best_model
      )

      predicted=best_model.predict(X_test)
      r2=r2_score(Y_test,predicted)
      return r2
    

    except Exception as e:
       raise CustomException(e,sys)
      
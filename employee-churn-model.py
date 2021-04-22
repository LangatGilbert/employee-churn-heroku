import pandas as pd
import numpy
from pycaret.classification import setup,create_model,tune_model,save_model

train_data = pd.read_csv("../data/HR_training_data.csv")

#initializing pycaret environment
employee_class = setup(data = train_data, target = 'left', session_id=123)


#creating model
lightgbm = create_model('lightgbm')

#tuned the model by optimizing on AUC
tuned_lightgbm = tune_model(lightgbm, optimize = 'AUC')

#saving the model
save_model(tuned_lightgbm,'../model/employees_churn_model')
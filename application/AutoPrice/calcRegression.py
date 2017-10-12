import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn import model_selection, preprocessing, feature_selection
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
from sklearn.dummy import DummyRegressor

def myModel(userVClass):
  print ""
  print "**********************************************************"
  print "                ",userVClass
  print "**********************************************************"
  dataFile_sedan="data/VehicleDataCleanSorted_SedanMid.csv"
  dataFile_truck="data/VehicleDataCleanSorted_Truck.csv"
  dataFile_SUV="data/VehicleDataCleanSorted_SUV.csv"
  data_sedan_df=pd.read_csv(dataFile_sedan)
  data_truck_df=pd.read_csv(dataFile_truck)
  data_SUV_df=pd.read_csv(dataFile_SUV)

  ##  Clean up dataFile to inlcude only columns we need and store as PandasDF
  keep_col = ['StickerPrice','comb08','displ','TCO']
  coef_dict={'c1':'MSRP', 'c2':'MPG', 'c3':'Disp' }
  data_sedan_df=data_sedan_df[keep_col]
  data_truck_df=data_truck_df[keep_col]
  data_SUV_df=data_SUV_df[keep_col]

  ## Choose appropriate datafile
  if userVClass == "sedan":
    data_df=data_sedan_df
  if userVClass == "truck":
    data_df=data_truck_df
  if userVClass == "SUV":
      data_df=data_SUV_df


  ##  Create numpy array for model building
  npMatrix=np.matrix(data_df)
  X = npMatrix[:,0:2]
  Y = npMatrix[:,3]

  ## Split the set into testing and training samples

  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)


  #min_max_scaler = preprocessing.MinMaxScaler()
  #X_train = min_max_scaler.fit_transform(X_train)
  #X_test = min_max_scaler.fit_transform(X_test)
  #Y_train = min_max_scaler.fit_transform(Y_train)
  #Y_test = min_max_scaler.fit_transform(Y_test)

  ## Run the model
  model = LinearRegression()
  model.fit(X_train,Y_train)
  Y_pred = model.predict(X_test)
  Y_pred_full = model.predict(X)
  
  ## Dummy Model
  model_dummy=DummyRegressor(strategy='mean')
  model_dummy.fit(X_train,Y_train)
  Y_pred_dummy = model_dummy.predict(X_test)
  MAE = mean_absolute_error(Y_test, Y_pred_dummy)
  MSE = mean_squared_error(Y_test, Y_pred_dummy)
  EVS = explained_variance_score(Y_test, Y_pred_dummy)
  Score= model.score(X_test,Y_test)
  print "Dummy Regressor Scores:"
  print("MSE: %.2f" % MSE )
  print("Mean absolute error: %.2f" % MAE)
  print("Explained Variance Score: %.2f" % EVS)
  print("Score: %.2f" % Score)
  print ""

  ## Regression Coefficients
  print "Regression Coefficients:"
  coef={}
  for i, value in enumerate(model.coef_[0]):
    name="c"+str(i+1)
    coef[name]=value
  for key, value in coef.items():
    print("%s (%s): %.2f") % (key, coef_dict[key],value)
  b = model.intercept_[0]
  print "b: ", b
  print ""

  ## Validation metrics
  print "K-Fold Validation Scores: mean (std)"
  scoring_metrics = {'MeanAE':'neg_mean_absolute_error','MSE':'neg_mean_squared_error','MedAR':'neg_median_absolute_error','R2':'r2'}
  kfold = model_selection.KFold(n_splits=5, random_state=0)
  for key, value in scoring_metrics.items():
    results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=value)
    print("%s: %.3f (%.3f)") % (key, results.mean(), results.std())
  print ""


  ## Test data scores
  print "Test Data Scores:"
  MAE = mean_absolute_error(Y_test, Y_pred)
  MSE = mean_squared_error(Y_test, Y_pred)
  EVS = explained_variance_score(Y_test, Y_pred)
  Score= model.score(X_test,Y_test)
  print("MSE: %.2f" % MSE )
  print("Mean absolute error: %.2f" % MAE)
  print("Explained Variance Score: %.2f" % EVS)
  print("Score: %.2f" % Score)
  print ""

myModel('sedan')
myModel('truck')
myModel('SUV')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
from sklearn import model_selection
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, accuracy_score
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split

def myModel(userVClass):

  dataFile_sedan="/Users/andrewchegwidden/Desktop/Insight/application/AutoPrice/data/VehicleDataCleanSorted_SedanMid.csv"
  dataFile_truck="/Users/andrewchegwidden/Desktop/Insight/application/AutoPrice/data/VehicleDataCleanSorted_Truck.csv"
  dataFile_SUV="/Users/andrewchegwidden/Desktop/Insight/application/AutoPrice/data/VehicleDataCleanSorted_SUV.csv"
  data_sedan_df=pd.read_csv(dataFile_sedan)
  data_truck_df=pd.read_csv(dataFile_truck)
  #data_suv_df=pd.read_csv(dataFile_suv)
  ##  Clean up dataFile to inlcude only columns we need and store as PandasDF


  keep_col = ['StickerPrice','comb08','displ','TCO']
  data_sedan2_df=data_sedan_df[keep_col]
  data_truck2_df=data_truck_df[keep_col]
  #data_suv2_df=data_suv_df[keep_col]

  if userVClass == "sedan":
    data_df=data_sedan2_df
    data2_df=data_truck2_df
    #data3_df=data_suv2_df
  if userVClass == "truck":
    data_df=data_truck2_df
    data2_df=data_sedan2_df
    #data3_df=data_suv2_df
#  if userVClass == "suv":
#    data_df=data_suv2_df
#    data2_df=data_sedan2_df
#    data3_df=data_truck2_df

  ##  Create numpy matrix for model building
  npmatrix=np.matrix(data_df)
  npmatrix2=np.matrix(data2_df)
  #npmatrix3=np.matrix(data3_df)

  X = npmatrix[:,0]
  Y = npmatrix[:,3]
  X2 = npmatrix2[:,0]
  Y2 = npmatrix2[:,3]
  #X3 = npmatrix3[:,0:2]
  #Y3 = npmatrix3[:,3]



  ## Split the set into testing and training samples
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25,random_state=42)


  

  print "asdf"
#print f_regression(X_train,Y_train)

  ## Run the model
  model = LinearRegression()
  model.fit(X_train,Y_train)
  Y_pred = model.predict(X_test)
  Y_pred_full = model.predict(X)


  ## Validation metrics
  print "k-fold validation:"
  scoring_metrics = {'MeanAE':'neg_mean_absolute_error','MSE':'neg_mean_squared_error','MedAR':'neg_median_absolute_error','R2':'r2'}
  kfold = model_selection.KFold(n_splits=4, random_state=7)
  for key, value in scoring_metrics.items():
    results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=value)
    print("%s: %.3f (%.3f)") % (key, results.mean(), results.std())
  print ""

  ## Regression Coefficients
  print "Regression Coefficients:"
  coef={}
  for i, value in enumerate(model.coef_[0]):
    name="c"+str(i+1)
    coef[name]=value
  coef['b']= model.intercept_[0]
  print coef
  print ""

  MAE = mean_absolute_error(Y_test, Y_pred)
  MSE = mean_squared_error(Y_test, Y_pred)
  EVS = explained_variance_score(Y_test, Y_pred)
  R2 = r2_score(Y_test,Y_pred)
  R2_2= model.score(X_train,Y_train)
  Score=model.score(X_test,Y_test)
  print("Mean squared error: %.2f" % MSE )
  print("Mean absolute error: %.2f" % MAE)
  print("Explained Variance Score: %.2f" % EVS)
  print("RSquared: %.2f" % R2)
  print("RSquared2: %.2f" % R2_2)
  print("Score: %.2f" % Score)
  print ""








myModel("truck")


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LinearRegression, Lasso, HuberRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, accuracy_score
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split
matplotlib.style.use('ggplot')
from sklearn.svm import SVR

def myModel(userVClass):

  dataFile="/Users/andrewchegwidden/Desktop/Insight/application/AutoPrice/data/VehicleDataCleanSorted_"+userVClass+".csv"
  data_df2=pd.read_csv(dataFile)
  
  ##  Clean up dataFile to inlcude only columns we need and store as PandasDF
  keep_col = ['StickerPrice','comb08','displ','TCO']
  data_df=data_df2[keep_col]

  ##  Create numpy array for model building
  #npmatrix=np.matrix(data_df)
  #X = npmatrix[:,0:2]
  #Y = npmatrix[:,3]
  X=data_df[['StickerPrice','comb08']]
  Y=data_df[['TCO']]
  
  
  X_sticker=data_df[['StickerPrice']]
  X_mpg=data_df[['comb08']]
  
  ## Split the set into testing and training samples
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.50, random_state=42)


  ## Declaring Model Types
  model0=LinearRegression()
  model1=SVR(kernel='linear')
  model2=HuberRegressor()
  #models_str = ['LinearRegression','SVR','HuberRegressor']
  models_str = ['LinearRegression','SVR']

  ## Declaring dicts
  #models = {models_str[0]:model0, models_str[1]:model1, models_str[2]:model2}
  models = {models_str[0]:model0, models_str[1]:model1}
  modelScores={}
  modelCoefs={}
  modelIntercept={}
  modelMSE={}
  modelY_pred={}
  modelY_pred_full={}
  modelY_test={}
  modelY={}
  modelX={}

  ## Loop over models and fill dicts
  for i, model_str in enumerate(models_str):
    models[model_str].fit(X_train,Y_train)
    Y_pred = models[model_str].predict(X_test)
    Y_pred_full = models[model_str].predict(X)
    modelY_pred[model_str]=Y_pred
    modelY_pred_full[model_str]=Y_pred_full
    modelY_test[model_str]=Y_test
    modelY[model_str]=Y
    modelX[model_str]=X
    score=models[model_str].score(X_test,Y_test)
    MSE = mean_squared_error(Y_test, Y_pred)
    if 'Huber' not in model_str:
      modelIntercept[model_str]=models[model_str].intercept_[0]
      modelCoefs[model_str]=models[model_str].coef_[0]
    modelScores[model_str]=score
    modelMSE[models_str[i]]=MSE

  ## Loop over models and print metrics
  for i, model_str in enumerate(models_str):
    print model_str
    print 'R2:   ',modelScores[model_str]
    print 'MSE:   ',modelMSE[model_str]
    if 'Huber' not in model_str:
      print 'b:    ',modelIntercept[model_str]
      print 'coefs: ',modelCoefs[model_str]
    print ''

  ## K-fold validation metrics
  print "k-fold validation:"
  scoring_metrics = {'MeanAE':'neg_mean_absolute_error','MSE':'neg_mean_squared_error','MedAR':'neg_median_absolute_error','R2':'r2'}
  kfold = model_selection.KFold(n_splits=4, random_state=7)
  for i, model_str in enumerate(models_str):
    print model_str
    for key, value in scoring_metrics.items():
      results = model_selection.cross_val_score(models[model_str], X, Y, cv=kfold, scoring=value)
      print("%s: %.3f (%.3f)") % (key, results.mean(), results.std())
    print ''

  ## Plot True vs Pred
  fig = plt.figure()
  ax1 = fig.add_subplot(111)
  ax1.set_xlabel('True Value')
  ax1.set_ylabel('Predicted Value')
  for i, model_str in enumerate(models_str):
    ax1.scatter(modelY[model_str],modelY_pred_full[model_str],label=(model_str+"  R2: {0:.2f}".format(modelScores[model_str])))
    plt.legend(loc='upper left')
  #plt.show()


  def PlotVariables(xvar='', yvar='', xaxis_label='', yaxis_label=''):
    dataFile_sedan="/Users/andrewchegwidden/Desktop/Insight/application/AutoPrice/data/VehicleDataCleanSorted_SedanMid.csv"
    dataFile_truck="/Users/andrewchegwidden/Desktop/Insight/application/AutoPrice/data/VehicleDataCleanSorted_Truck.csv"
    data_df_sedan=pd.read_csv(dataFile_sedan)
    data_df_truck=pd.read_csv(dataFile_truck)

                        
    X_sedan=data_df_sedan[[xvar]]
    X_truck=data_df_truck[[xvar]]

    Y_sedan=data_df_sedan[[yvar]]
    Y_truck=data_df_truck[[yvar]]

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.set_xlabel(xaxis_label)
    ax2.set_ylabel(yaxis_label)
    ax2.scatter(X_sedan,Y_sedan,label='Sedan')
    ax2.scatter(X_truck,Y_truck,label='Truck')
    plt.legend(loc='upper left')
    #plt.show()
    fig2.savefig(xvar+'_vs_'+yvar+'.png')

  PlotVariables(xvar='StickerPrice', yvar='TCO', xaxis_label='Sticker Price [$]',yaxis_label='Target Value [$]')
  PlotVariables(xvar='comb08', yvar='TCO', xaxis_label='Vehicle MPG',yaxis_label='Target Value [$]')
  PlotVariables(xvar='displ', yvar='TCO', xaxis_label='Engine Displacement [l]',yaxis_label='Target Value [$]')

## 'StickerPrice'   'comb08'   'displ'   'TOC
myModel("SedanMid")

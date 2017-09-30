import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LinearRegression, Lasso, HuberRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, accuracy_score, f1_score
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.svm import SVR
from sklearn.datasets import make_regression





def mean_absolute_percentage_error(y_true, y_pred):
  return np.mean(np.abs((y_true - y_pred) / y_true))


def myModel(userVClass):
  print ""
  print "**********************************************************"
  print "                ",userVClass
  print "**********************************************************"

  dataFile="/Users/andrewchegwidden/Desktop/Insight/application/AutoPrice/data/VehicleDataCleanSorted_"+userVClass+".csv"
  data_df2=pd.read_csv(dataFile)
  
  ##  Clean up dataFile to inlcude only columns we need and store as PandasDF
  keep_col = ['StickerPrice','comb08','displ','TCO']
  data_df=data_df2[keep_col]

  ##  Create numpy array for model building
  npmatrix=np.matrix(data_df)
  X=data_df[['StickerPrice','comb08']]
  Y=data_df[['TCO']]
  
  
  X_sticker=data_df[['StickerPrice']]
  X_mpg=data_df[['comb08']]
  
  ## Split the set into testing and training samples
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)


  ## Declaring Model Types
  
  
  models_str = ['LinearRegression','Lasso']; model0=LinearRegression(); model1=Lasso(alpha=100.0); models = {models_str[0]:model0,models_str[1]:model1}
  #models_str = ['Lasso']; model0=Lasso(alpha=1000); models = {models_str[0]:model0}

  ## Declaring dicts
  modelScores={}
  modelCoefs={}
  modelIntercept={}
  modelMSE={}
  modelMAE={}
  modelMAE_full={}
  modelMAPE={}
  modelMAPE_full={}
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
    MAE = mean_absolute_error(Y_test, Y_pred)
    MAE_full = mean_absolute_error(Y, Y_pred_full)
    #MAPE= mean_absolute_percentage_error(Y_test, Y_pred)
    #MAPE_full= mean_absolute_percentage_error(Y, Y_pred_full)
    modelIntercept[model_str]=models[model_str].intercept_[0]
    modelCoefs[model_str]=models[model_str].coef_
    modelScores[model_str]=score
    modelMSE[models_str[i]]=MSE
    modelMAE[models_str[i]]=MAE
    modelMAE_full[models_str[i]]=MAE_full
    #modelMAPE[models_str[i]]=MAPE
    #modelMAPE_full[models_str[i]]=MAPE_full

  ## Loop over models and print metrics
  for i, model_str in enumerate(models_str):
    print model_str
    print 'R2:    ',modelScores[model_str]
    print 'MSE:   ',modelMSE[model_str]
    print 'MAE:   ',modelMAE[model_str]
    print 'MAE_f: ',modelMAE_full[model_str]
    #print 'MAPE:   ',modelMAPE[model_str]
    #print 'MAPE_f: ',modelMAPE_full[model_str]
    print 'b:     ',modelIntercept[model_str]
    print 'coefs: ',modelCoefs[model_str]
    print ''

  ## K-fold validation metrics
  print "k-fold validation:"
  scoring_metrics = {'MeanAE':'neg_mean_absolute_error','MSE':'neg_mean_squared_error','MedAR':'neg_median_absolute_error','R2':'r2'}
  kfold = model_selection.KFold(n_splits=5, random_state=7)
  for i, model_str in enumerate(models_str):
    print model_str
    for key, value in scoring_metrics.items():
      results = model_selection.cross_val_score(models[model_str], X, Y, cv=kfold, scoring=value)
      print("%s: %.3f (%.3f)") % (key, results.mean(), results.std())
    print ''

  ## Plot kFold Pred vs True
  for i, model_str in enumerate(models_str):
    predicted = cross_val_predict(models[model_str],X,Y,cv=5)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(Y,predicted,edgecolors=(0,0,0))
    ax.set_xlabel('True Value [$]',fontsize=15)
    ax.set_ylabel('Predicted Value [$]',fontsize=15)
    if userVClass == 'Truck':
      plt.plot([32500,52000],[32500,52000], color = 'green')
    elif userVClass == 'SedanMid':
      plt.plot([25000,100000],[25000,100000], color = 'green')
    elif userVClass == 'SUV':
      plt.plot([30000,120000],[30000,120000], color = 'green')
    plt.title('5-Fold Cross Validation',fontsize=15)
    fig.savefig('Plots/'+userVClass+'_kfold_Pred_vs_True_'+model_str+'_withline.png',bbox_inches='tight')
    plt.close('all')

  ## Plot model param True vs Pred
  for i, model_str in enumerate(models_str):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('True Value')
    ax.set_ylabel('Predicted Value')
    ax.scatter(modelY[model_str]/60.0,modelY_pred_full[model_str]/60.0,label=(model_str+"  R2: {0:.2f}".format(modelScores[model_str])),edgecolors=(0,0,0))
    plt.legend(loc='upper left')
    if userVClass == 'Truck':
      plt.plot([32500/60.0,52000/60.0],[32500/60.0,52000/60.0], color = 'green')
    elif userVClass == 'SedanMid':
      plt.plot([25000/60.0,100000/60.0],[25000/60.0,100000/60.0], color = 'green')
    elif userVClass == 'SUV':
      plt.plot([30000/60.0,120000/60.0],[30000/60.0,120000/60.0], color = 'green')
    fig.savefig('Plots/'+userVClass+'_Pred_vs_True_'+model_str+'_withline.png',bbox_inches='tight')
    plt.close('all')

  ## Model independnet plots
  def PlotVariables(userVClass, xvar='', yvar='', xaxis_label='', yaxis_label=''):
    dataFile_sedan="/Users/andrewchegwidden/Desktop/Insight/application/AutoPrice/data/VehicleDataCleanSorted_SedanMid.csv"
    dataFile_truck="/Users/andrewchegwidden/Desktop/Insight/application/AutoPrice/data/VehicleDataCleanSorted_Truck.csv"
    dataFile_suv="/Users/andrewchegwidden/Desktop/Insight/application/AutoPrice/data/VehicleDataCleanSorted_SUV.csv"
    data_df_sedan=pd.read_csv(dataFile_sedan)
    data_df_truck=pd.read_csv(dataFile_truck)
    data_df_suv=pd.read_csv(dataFile_suv)
    
    total = [data_df_sedan,data_df_truck,data_df_suv]
    data_df_total = pd.concat(total)
    
    X_sedan=data_df_sedan[[xvar]]
    X_truck=data_df_truck[[xvar]]
    X_suv=data_df_suv[[xvar]]
    X_total=data_df_total[[xvar]]

    Y_sedan=data_df_sedan[[yvar]]
    Y_truck=data_df_truck[[yvar]]
    Y_suv=data_df_suv[[yvar]]
    Y_total=data_df_total[[yvar]]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(xaxis_label, fontsize=15)
    ax.set_ylabel(yaxis_label, fontsize=15)
    ax.scatter(X_sedan,Y_sedan,label='Sedan',edgecolors=(0,0,0))
    ax.scatter(X_truck,Y_truck,label='Truck',edgecolors=(0,0,0))
    ax.scatter(X_suv,Y_suv,label='SUV',edgecolors=(0,0,0))
    plt.legend(loc='upper left')
    fig.savefig('Plots/'+xvar+'_vs_'+yvar+'.png',bbox_inches='tight')
    if xvar == 'StickerPrice' and yvar == 'TCO':
      plt.plot([20000,100000],[20000,100000], color = 'green')
      fig.savefig('Plots/'+xvar+'_vs_'+yvar+'_withline.png',bbox_inches='tight')
    plt.close('all')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(xaxis_label, fontsize=15)
    ax.set_ylabel(yaxis_label, fontsize=15)
    ax.scatter(X_total,Y_total,label='All Vehicles',edgecolors=(0,0,0))
    plt.legend(loc='upper left')
    fig.savefig('Plots/All_'+xvar+'_vs_'+yvar+'_.png',bbox_inches='tight')
    if xvar == 'StickerPrice' and yvar == 'TCO':
      plt.plot([20000,100000],[20000,100000], color = 'green')
      fig.savefig('Plots/All_'+xvar+'_vs_'+yvar+'_withline.png',bbox_inches='tight')
    plt.close('all')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(xaxis_label, fontsize=15)
    ax.set_ylabel(yaxis_label, fontsize=15)
    if userVClass == "SedanMid":
      ax.scatter(X_sedan,Y_sedan,label='Sedan',edgecolors=(0,0,0))
    elif userVClass == "Truck":
      ax.scatter(X_truck,Y_truck,label='Truck',edgecolors=(0,0,0))
    elif userVClass == "SUV":
      ax.scatter(X_suv,Y_suv,label='SUV',edgecolors=(0,0,0))
    fig.savefig('Plots/'+userVClass+'_'+xvar+'_vs_'+yvar+'_'+model_str+'.png',bbox_inches='tight')
    if xvar == 'StickerPrice' and yvar == 'TCO':
      plt.plot([20000,100000],[20000,100000], color = 'green')
    plt.legend(loc='upper left')
    fig.savefig('Plots/'+userVClass+'_'+xvar+'_vs_'+yvar+'_'+model_str+'_withline.png',bbox_inches='tight')
    plt.close('all')

  PlotVariables(userVClass=userVClass,xvar='StickerPrice', yvar='TCO', xaxis_label='MSRP [$]',yaxis_label='Total Cost of Ownership [$]')
  PlotVariables(userVClass=userVClass,xvar='comb08', yvar='TCO', xaxis_label='Vehicle MPG',yaxis_label='Total Cost of Ownership [$]')
  PlotVariables(userVClass=userVClass,xvar='displ', yvar='TCO', xaxis_label='Engine Displacement [L]',yaxis_label='Total Cost of Ownership [$]')
  PlotVariables(userVClass=userVClass,xvar='StickerPrice', yvar='comb08', xaxis_label='MSRP [$]',yaxis_label='Vehicle MPG')







myModel("Truck")
myModel("SUV")
myModel("SedanMid")



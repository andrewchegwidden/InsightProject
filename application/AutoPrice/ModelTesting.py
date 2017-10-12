import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn import model_selection, preprocessing
from sklearn.linear_model import LinearRegression, Lasso, HuberRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, accuracy_score, f1_score
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.model_selection import train_test_split, cross_val_predict





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

  ##  Create PandasDF array for model building
  X=data_df[['StickerPrice','displ','comb08']]
  Y=data_df[['TCO']]
  
  
  X_sticker=data_df[['StickerPrice']]
  X_mpg=data_df[['comb08']]
  X_displ=data_df[['displ']]
  
  

  
  ## Split the set into testing and training samples
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

  ##
  #scaler_x = preprocessing.MinMaxScaler()
  #scaler_x.fit(X_train)
  #scaler_y = preprocessing.MinMaxScaler()
  #scaler_y.fit(Y_train)
  #X_train = scaler_x.transform(X_train)
  #X_test = scaler_x.transform(X_test)
  #Y_train = scaler_y.transform(Y_train)
  #Y_test = scaler_y.transform(Y_test)


  model = SelectKBest(score_func=f_regression,k=1)
  results = model.fit(X_train,Y_train)
  print 'scores: ',results.scores_
  print 'pvalues: ',results.pvalues_


  ## Declaring dicts
  modelScores_test={}
  modelScores_train={}
  modelScores_full={}

  modelCoefs={}
  modelIntercept={}

  modelMAE_test={}
  modelMAE_train={}
  modelMAE_full={}

  modelMAPE_test={}
  modelMAPE_train={}
  modelMAPE_full={}

  modelY_pred_test={}
  modelY_pred_train={}
  modelY_pred_full={}

  modelY_test={}
  modelY_train={}
  modelY_full={}
  modelX_test={}
  modelX_train={}
  modelX_full={}
  
  modelFeatureScores_test={}
  modelFeatureScores_train={}
  modelFeatureScores_full={}

  ## Declaring Model Types
  #models_str = ['Lasso']; model0=Lasso(alpha=0.000000001); models = {models_str[0]:model0}
  models_str = ['LinearRegression']; model0=LinearRegression(); models = {models_str[0]:model0}

  ## Loop over models and fill dicts
  for i, model_str in enumerate(models_str):
    models[model_str].fit(X_train,Y_train)
    
    Y_pred_test = models[model_str].predict(X_test)
    Y_pred_train = models[model_str].predict(X_train)
    Y_pred_full = models[model_str].predict(X)

    modelY_pred_test[model_str]=Y_pred_test
    modelY_pred_train[model_str]=Y_pred_train
    modelY_pred_full[model_str]=Y_pred_full
 
    modelY_test[model_str]=Y_test
    modelY_train[model_str]=Y_train
    modelY_full[model_str]=Y

    modelX_test[model_str]=X_test
    modelX_train[model_str]=X_train
    modelX_full[model_str]=X
    
    score_test=models[model_str].score(X_test,Y_test)
    score_train=models[model_str].score(X_train,Y_train)
    score_full=models[model_str].score(X,Y)

    MAE_test = mean_absolute_error(Y_test, Y_pred_test)
    MAE_train = mean_absolute_error(Y_train, Y_pred_train)
    MAE_full = mean_absolute_error(Y, Y_pred_full)

    MAPE_test= mean_absolute_percentage_error(Y_test, Y_pred_test)
    MAPE_train= mean_absolute_percentage_error(Y_train, Y_pred_train)
    MAPE_full= mean_absolute_percentage_error(Y, Y_pred_full)
    
    FeatureScore_test=SelectKBest(score_func=f_regression ,k=3);
    FeatureScore_train=SelectKBest(score_func=f_regression ,k=3);
    FeatureScore_full=SelectKBest(score_func=f_regression ,k=3);

    modelIntercept[model_str]=models[model_str].intercept_[0]
    modelCoefs[model_str]=models[model_str].coef_
    modelScores_test[model_str]=score_test
    modelScores_train[model_str]=score_train
    modelScores_full[model_str]=score_full

    modelMAE_test[models_str[i]]=MAE_test
    modelMAE_train[models_str[i]]=MAE_train
    modelMAE_full[models_str[i]]=MAE_full
    
    modelMAPE_test[models_str[i]]=MAPE_test
    modelMAPE_train[models_str[i]]=MAPE_train
    modelMAPE_full[models_str[i]]=MAPE_full

    modelFeatureScores_test[models_str[i]]=FeatureScore_test.fit(X_test,Y_test)
    modelFeatureScores_train[models_str[i]]=FeatureScore_train.fit(X_train,Y_train)
    modelFeatureScores_full[models_str[i]]=FeatureScore_full.fit(X,Y)

## Loop over models and print metrics
  for i, model_str in enumerate(models_str):
    print model_str
    print "Metric: (train) (test) (full)"
    print("R2: (%.3f) (%.3f) (%.3f)") % (modelScores_train[model_str], modelScores_test[model_str], modelScores_full[model_str])
    print("MAE: (%.3f) (%.3f) (%.3f)") % (modelMAE_train[model_str]/60.0, modelMAE_test[model_str]/60.0, modelMAE_full[model_str]/60.0)
    #print("MAPE: (%.3f) (%.3f) (%.3f)") % (modelMAPE_train[model_str], modelMAPE_test[model_str], modelMAPE_full[model_str])
    #print("FeatScore: (%.3f) (%.3f) (%.3f)") % (modelFeatureScores_train[model_str], modelFeatureScores_test[model_str], modelFeatureScores_full[model_str])
    print "FeatScore: (",modelFeatureScores_train[model_str].scores_,") (",modelFeatureScores_test[model_str].scores_,")"

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
    ax.scatter(modelY_full[model_str]/60.0,modelY_pred_full[model_str]/60.0,label=(model_str+"  R2: {0:.2f}".format(modelScores_full[model_str])),edgecolors=(0,0,0))
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
  PlotVariables(userVClass=userVClass,xvar='comb08', yvar='displ', xaxis_label='Vehicle MPG',yaxis_label='Engine Displacement [L]')
  PlotVariables(userVClass=userVClass,xvar='StickerPrice', yvar='displ', xaxis_label='MSRP [$]',yaxis_label='Engine Displacement [L]')







#myModel("Truck")
#myModel("SUV")
myModel("SedanMid")



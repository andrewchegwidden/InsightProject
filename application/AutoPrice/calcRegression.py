import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.model_selection import train_test_split

def myModel(userVClass):

  dataFile_sedan="AutoPrice/data/VehicleDataCleanSorted_SedanMid.csv"
  dataFile_truck="AutoPrice/data/VehicleDataCleanSorted_Truck.csv"
  data_sedan_df=pd.read_csv(dataFile_sedan)
  data_truck_df=pd.read_csv(dataFile_truck)
  
  ##  Clean up dataFile to inlcude only columns we need and store as PandasDF


  keep_col = ['StickerPrice','comb08','displ','TCO']
  data_sedan2_df=data_sedan_df[keep_col]
  data_truck2_df=data_truck_df[keep_col]

  if userVClass == "sedan":
    data_df=data_sedan2_df
  if userVClass == "truck":
    data_df=data_truck2_df



  ##  Create numpy array for model building
  npMatrix=np.matrix(data_df)
  X = npMatrix[:,0]
  Y = npMatrix[:,3]

  ## Split the set into testing and training samples
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)


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
  b = model.intercept_[0]
  print coef
  print ""

  MAE = mean_absolute_error(Y_test, Y_pred)
  MSE = mean_squared_error(Y_test, Y_pred)
  EVS = explained_variance_score(Y_test, Y_pred)
  R2 = r2_score(Y_test,Y_pred)
  Score= model.score(X_train,Y_train)
  print("Mean squared error: %.2f" % MSE )
  print("Mean absolute error: %.2f" % MAE)
  print("Explained Variance Score: %.2f" % EVS)
  print("RSquared: %.2f" % R2)
  print("Score: %.2f" % Score)
  print ""

  ## Plotting (only used locally)
#  mydf1=pd.DataFrame.from_records(Y_pred_full)
#  mydf1=mydf1.rename(columns = {0:'Predicted TCO'})
#  mydf2=pd.DataFrame.from_records(Y.tolist())
#  mydf2=mydf2.rename(columns = {0:'TCO'})
#  PredTrue_df=mydf1.join(mydf2)

#  mydf3=pd.DataFrame.from_records(Y_pred_full*1.2)
#  mydf3=mydf3.rename(columns = {0:'Predicted TCO2'})
#  PredTrue_df2=PredTrue_df.join(mydf3)


#  ax = data_sedan2_df.plot.scatter(x='StickerPrice',y='TCO', color='Red', label='Sedan')
#  data_truck2_df.plot.scatter(x='StickerPrice',y='TCO', color='Green', label='Truck', ax=ax)

#  plt.show()





#Y_pred_full=model.predict(X)
#Y_pred_full_df = pd.DataFrame(Y_pred_full)

 #data_df.plot.scatter(x='StickerPrice',y='TCO',c=data_df['StickerPrice']*coef['c1'] + data_df['comb08']*coef['c2'] + data_df['displ']*coef['c3'] + b, s=50)
 #plt.show()


  #data_df.plot.scatter(x='TCO',y='TCO',c=data_df['StickerPrice']*coef['c1'] + data_df['comb08']*coef['c2'] + data_df['displ']*coef['c3'] + b, s=50)
  #plt.show()

#pd.tools.plotting.scatter_matrix(data_df.loc[:,"StickerPrice":"displ"], diagonal="kde")
#  plt.tight_layout()
#  plt.show()


  print "30k: " , (model.predict(30000)/60.0)

  results = []
  results = [model.coef_[0][0], model.intercept_[0] ]

  return results



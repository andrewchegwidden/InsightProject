import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')


def myModel(userVClass):
  print ""
  print "**********************************************************"
  print "                ",userVClass
  print "**********************************************************"

  dataFile="/Users/andrewchegwidden/Desktop/Insight/application/AutoPrice/data/VehicleDataCleanSorted_"+userVClass+".csv"
  data_df2=pd.read_csv(dataFile)
  
  ##  Clean up dataFile to inlcude only columns we need and store as PandasDF
  keep_col = ['StickerPrice','comb08','displ','cylinders','TCO']
  keep_col2 = ['MSRP','MPG','EngDispl','Cylinders','TargetValue']
  
  
  data_df=data_df2[keep_col]
  

  correlations = data_df.corr()
  # plot correlation matrix
  fig = plt.figure()
  ax = fig.add_subplot(111)
  cax = ax.matshow(correlations, vmin=-1, vmax=1)
  fig.colorbar(cax)
  ticks = np.arange(0,7,1)
  ax.set_xticks()
  ax.set_yticks(ticks)
  ax.set_xticklabels(keep_col2)
  ax.set_yticklabels(keep_col2)
  fig.savefig('CorrelationPlot_'+userVClass+'.png')


myModel("SUV")
myModel("Truck")
myModel("SedanMid")



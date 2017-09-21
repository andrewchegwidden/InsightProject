from flask import render_template
from AutoPrice import app_AutoPrice
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2
from flask import request
from calcRegression import myModel

#user = 'andrewchegwidden' #add your username here (same as previous postgreSQL)
#host = 'localhost'
#dbname = 'birth_db'
#db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
#con = None
#con = psycopg2.connect(database = dbname, user = user)

@app_AutoPrice.route('/')

@app_AutoPrice.route('/home')
def home():
    return render_template("home.html")


@app_AutoPrice.route('/home2')
def home2():
  global userVClass
  global userMethod
  userVClass = request.args.get('userVClass')
  userMethod = request.args.get('userMethod')
  if userVClass.lower() == "suv":
    userVClass = "SUV"
  if "car" in userVClass.lower():
    userVClass = "sedan"
  if "sedan" in userVClass.lower():
    userVClass = "sedan"
  if "truck" in userVClass.lower():
    userVClass = "truck"
  if "suv" in userVClass.lower():
    userVClass = "SUV"
  if "sticker" in userMethod.lower():
    userMethod = "MSRP"
  if "budget" in userMethod.lower():
    userMethod="Budget"
  print "userVClass: ",userVClass
  print "userMethod: ",userMethod
  if userMethod == "Budget":
    return render_template("home2_Budget.html",userVClass=userVClass,userMethod=userMethod)
  if userMethod == "MSRP":
    return render_template("home2_MSRP.html",userVClass=userVClass,userMethod=userMethod)

@app_AutoPrice.route('/home3')
def home3():
  print "hi"


@app_AutoPrice.route('/htw')
def htw():
    return render_template("htw.html")

@app_AutoPrice.route('/output')
def output():
    #mycoefs=myModel(userVClass)
    mycoefs = []
    if userVClass == "sedan":
      mycoefs = [0.968324198962,12560.0547686]
    if userVClass == "truck":
      mycoefs = [0.657331882067,19131.8760612]
    userMSRP = request.args.get('userMSRP')
    userMSRP_str= '{:,.0f}'.format(float(userMSRP))
    if not mycoefs:
      return render_template("output.html", the_result = "(error)", the_result_error="(error)", userMSRP=userMSRP, userMSRP_str=userMSRP_str, userVClass=userVClass)
    else:
      the_result=int( (mycoefs[0]*float(userMSRP) + mycoefs[1]) / 60.0 )
      print "This is the userVClass: ", userVClass
      print "This is the userMSRP: ", userMSRP
      print "This is c1: ",mycoefs[0]
      print "This is b:  ",mycoefs[1]
      print "This is MAE:  ",mycoefs[2]
      return render_template("output.html", the_result = the_result, userMSRP=userMSRP, userMSRP_str=userMSRP_str, userVClass=userVClass, c1=mycoefs[0], b=mycoefs[1])

@app_AutoPrice.route('/output2')
def output2():
    #userVClass = request.args.get('userVClass')
    #if userVClass.lower() == "suv":
    #    userVClass = "SUV"
    mycoefs=myModel(userVClass)
    userBudget = request.args.get('userBudget')
    userBudget_str= '{:,.0f}'.format(float(userBudget))
    if not mycoefs:
      return render_template("output2.html", userBudget=userBudget, userBudget_str=userBudget_str,userVClass=userVClass)
    else:
      the_result=int( (float(userBudget)*60.0 - mycoefs[1]) / mycoefs[0] )
      print "This is the userVClass: ", userVClass
      print "This is the userBudget: ", userBudget
      print "This is c1: ",mycoefs[0]
      print "This is b:  ",mycoefs[1]
      return render_template("output2.html", the_result = the_result, userBudget=userBudget, userBudget_str=userBudget_str,userVClass=userVClass)


@app_AutoPrice.route('/output')
def output():
  global userVClass
  global userMethod
  userVClass = request.args.get('userVClass')
  userMethod = request.args.get('userMethod')
  if userVClass.lower() == "suv":
    userVClass = "SUV"
  if "car" in userVClass.lower():
    userVClass = "sedan"
  if "sedan" in userVClass.lower():
    userVClass = "sedan"
  if "truck" in userVClass.lower():
    userVClass = "truck"
  if "suv" in userVClass.lower():
    userVClass = "SUV"
  if "sticker" in userMethod.lower():
    userMethod = "MSRP"
  if "budget" in userMethod.lower():
    userMethod="Budget"
  mycoefs = []
  print userVClass
  if userVClass == "sedan":
    mycoefs = [0.968324198962,12560.0547686]
  if userVClass == "truck":
    mycoefs = [0.657331882067,19131.8760612]
  if not mycoefs:
    return render_template("output3.html", the_result = "(error)", the_result_error="(error)", userVClass=userVClass)
  else:
    print "This is the userVClass: ", userVClass
    print "This is c1: ",mycoefs[0]
    print "This is b:  ",mycoefs[1]
    print "SHIT!!!!!!!: ", userMethod
    if userMethod == "Budget":
      return render_template("output_2.html", userVClass=userVClass, c1=mycoefs[0], b=mycoefs[1])
    if userMethod == "MSRP":
      return render_template("output_1.html", userVClass=userVClass, c1=mycoefs[0], b=mycoefs[1])





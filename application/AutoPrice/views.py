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


@app_AutoPrice.route('/htw')
def htw():
    return render_template("htw.html")


@app_AutoPrice.route('/output')
def output():
  global userVClass
  global userMethod
  userVClass = request.args.get('userVClass')
  userMethod = request.args.get('userMethod')
  if (userVClass is None) or (userMethod is None):
    return render_template("home.html")
  mycoefs = []
  if userVClass.lower() == "suv":
    userVClass = "SUV"
  if "car" in userVClass.lower():
    userVClass = "sedan"
  elif "sedan" in userVClass.lower():
    userVClass = "sedan"
    #mycoefs = [0.968324198962,12560.0547686]
    mycoefs = [0.92511604056627661,-242.6476549850758,20747.970831886545]
  elif "truck" in userVClass.lower():
    userVClass = "truck"
    #mycoefs = [0.657331882067,19131.8760612]
    mycoefs = [0.59657718843629359,-590.80535829169958,32561.359262109392]



  else:
    userVClass="Undefined"
  if "sticker" in userMethod.lower():
    userMethod = "MSRP"
  elif "budget" in userMethod.lower():
    userMethod="Budget"
  else:
    userMethod="Undefined"
  if (not mycoefs) or (userMethod == "Undefined") or (userVClass == "Undefined"):
    return render_template("ERROR3.html")
  else:
    print "This is the userVClass: ", userVClass
    print "This is c1: ",mycoefs[0]
    print "This is b:  ",mycoefs[1]
    print "SHIT!!!!!!!: ", userMethod
    if userMethod == "Budget":
      if userVClass == "sedan":
        return render_template("output_2_sedan.html", userVClass=userVClass, c1=mycoefs[0],c2=mycoefs[1], b=mycoefs[2])
      elif userVClass == "truck":
        return render_template("output_2_truck.html", userVClass=userVClass, c1=mycoefs[0],c2=mycoefs[1], b=mycoefs[2])
      else:
        return render_template("ERROR3.html")
    if userMethod == "MSRP":
      if userVClass == "sedan":
        return render_template("output_1_sedan.html", userVClass=userVClass, c1=mycoefs[0],c2=mycoefs[1], b=mycoefs[2])
      elif userVClass == "truck":
        return render_template("output_1_truck.html", userVClass=userVClass, c1=mycoefs[0],c2=mycoefs[1], b=mycoefs[2])
      else:
        return render_template("ERROR3.html")




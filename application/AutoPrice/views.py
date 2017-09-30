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

@app_AutoPrice.route('/about')
def about():
    return render_template("about.html")


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
  if "sedan" in userVClass.lower():
    userVClass = "sedan"
    mycoefs = [0.92511604056627661,-242.6476549850758,20747.970831886545]
    bsId="20211"
    mpg_init="25"
    calcnum=7200.0
  elif "truck" in userVClass.lower():
    userVClass = "truck"
    mycoefs = [0.7369752,-462.92720897,31660.3261817]
    bsId="20218"
    mpg_init="20"
    calcnum=9000.0
  elif "suv" in userVClass.lower():
    userVClass = "SUV"
    mycoefs = [0.94175931,-59.3097646, 13693.4734]
    bsId="20217"
    mpg_init="20"
    calcnum=9000.0
  else:
    userVClass="Undefined"

  if "msrp" in userMethod.lower():
    userMethod = "MSRP"
  elif "budget" in userMethod.lower():
    userMethod="Budget"
  else:
    userMethod="Undefined"
  if (not mycoefs) or (userMethod == "Undefined") or (userVClass == "Undefined"):
    return render_template("ERROR.html")
  else:
    print "This is the userVClass: ", userVClass
    print "This is the userMethod: ", userMethod
    print "This is c1: ",mycoefs[0]
    print "This is b:  ",mycoefs[1]
    if userMethod == "Budget" and (userVClass == "sedan" or userVClass == "truck" or userVClass == "SUV"):
      return render_template("output_2.html", userVClass=userVClass, c1=mycoefs[0],c2=mycoefs[1], b=mycoefs[2],bsId=bsId,mpg_init=mpg_init,calcnum=calcnum)
    elif userMethod == "MSRP" and (userVClass == "sedan" or userVClass == "truck" or userVClass == "SUV"):
      return render_template("output_1.html", userVClass=userVClass, c1=mycoefs[0],c2=mycoefs[1], b=mycoefs[2], bsId=bsId,mpg_init=mpg_init,calcnum=calcnum)
    else:
      return render_template("ERROR.html")



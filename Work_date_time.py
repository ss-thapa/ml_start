import pandas as pd
import numpy as np
pd.set_option("display.max_columns", None)
from datetime import datetime


date = pd.read_csv("/Users/sunilthapa/Desktop/My _projects/ML_intro/CSVs/orders.csv")

time = pd.read_csv("/Users/sunilthapa/Desktop/My _projects/ML_intro/CSVs/messages.csv")


date['date'] = pd.to_datetime(date['date'])


## extract year

date['year'] = date['date'].dt.year


## extract month

date['month'] = date['date'].dt.month

## extract month name

date['month_name'] = date['date'].dt.month_name()

## extract day 

date['day'] = date['date'].dt.day

## extract which day of the week is that

date['day_of_week'] = date['date'].dt.dayofweek

## extract day name

date['day_name'] = date['date'].dt.day_name()


## extract is weekend

date['day_is_weekend'] = np.where(date['day_name'].isin(['Sunday','Saturday']), 1,0)

## extract week of the year

date['week_of_year'] = date['date'].dt.isocalendar().week

## extract quarter 

date['quarter'] = date['date'].dt.quarter

## extract semester

date['semester'] = np.where(date['quarter'].isin([1,2]),1,2)


today = datetime.today().date()

## extract date time till now
date['time_till_now'] = (today - date['date'].dt.date)






## using time 

time['date'] = pd.to_datetime(time['date'])


time['hour'] = time['date'].dt.hour
time['min'] = time['date'].dt.minute
time['sec'] = time['date'].dt.second


## time difference 

(today - time['date'])/np.timedelta64(1,'h')




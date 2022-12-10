# -*- coding: utf-8 -*-

import pandas as pd
import spacy
import numpy as np
import pickle


df=pd.read_csv(".../Independent Project/Data/News data/20221030/yuncaijing_news_100days_yuncaijing_20221030.csv", 
               header=0, on_bad_lines='skip', usecols= ['datetime', 'title'])
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df = df.dropna(subset=['datetime'])

#df["datetime"] = pd.to_datetime(df.datetime,format='%Y-%m-%d')

#Change datetime to date
df["datetime"] = df['datetime'].dt.strftime('%Y-%m-%d')
df.rename(columns={'datetime':'date'},inplace=True)
df.to_csv (".../Independent Project/Data/News data/20221030/pre_yuncaijingnews_100days_20221030.csv", header=True, index =False)
df

sortdf=df.sort_values(['date'])
# makes all news of same date mapped to its corresponding date
data={}
corrpt={}

for i in range(len(sortdf)):
  temp=[]
  if(data.get(sortdf.iloc[i]['date'],None)==None):
    data[sortdf.iloc[i]['date']]=[]
  temp.append(str(sortdf.iloc[i]['title']).strip("\n"))
  try:
    if('http' not in sortdf.iloc[i]['url'] ):
      try:
        corrpt[sortdf.iloc[i]['date']].append(sortdf.iloc[i]['url'])
        temp.append(str(sortdf.iloc[i]['url']).strip("\n"))
      except:
        corrpt[sortdf.iloc[i]['date']]=[]
        corrpt[sortdf.iloc[i]['date']].append(sortdf.iloc[i]['url'].rstrip("\n"))
        temp.append(str(sortdf.iloc[i]['url']).strip("\n"))
  except Exception as e:
    print(e)
    print("No worries, Taken care before")
  data[sortdf.iloc[i]['date']].append(' '.join(temp))
for i in data:
  data[i]=set(data[i])
sortdf.iloc[1]['date']
len(sortdf)


#use "zh_core_web_sm" to do vector generation
model=spacy.load("zh_core_web_sm")
data_vec={}
count=0
for date in data:
    temp=[]
    #generates mean values for all the vectors
    for text in data[date]:
        doc=model(text)
        temp.append(doc.vector)
    #map the vector mean to that day
    meank=np.mean(np.array(temp,dtype=np.float32),axis=0)
    data_vec[date]=meank
    count+=1
    #print(count)
with open(".../Independent Project/Data/News data/20221030/"+"100days_yuncaijing20221030-vec.pkl","wb") as f:
    pickle.dump(data_vec,f)
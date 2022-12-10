# -*- coding: utf-8 -*-

import pandas as pd
import spacy
import numpy as np
import pickle


from os import set_blocking
names=["small"]
vectors={"small":None}
final_data={}
pricename={'small':'small'}
filepath = ".../Independent Project/Data/News data/20221030/100days_yuncaijing20221030-vec.pkl"
with open(filepath,"rb")as f:
  vectors["small"]=pickle.load(f)

df=pd.read_csv(".../Independent Project/Data/News data/20221030/pre_yuncaijingnews_100days_20221030.csv")
df2=pd.read_csv(".../Independent Project/Data/Stock data/100days_20221030/00175.csv")

for stock in df2.iloc[:1,1:]:
      final_data[stock]={"long":[],"medium":[],"short":[]}
      price=df2[stock]
      sortdf=df.sort_values(['date'])
      # makes all news of same date mapped to its corresponding date
      data={}
      corrpt={}
      #creates y train/test
      close_price=[]
      print("Setting up the close price for "+stock)
      for i in range(len(price)-1):
        close_price.append(1*(float(price.iloc[i+1])>0))
      print("Merging data for same date for "+stock)
      for i in range(len(sortdf)):
        temp=[]
        if(data.get(sortdf.iloc[i]['date'],None)==None):
            data[sortdf.iloc[i]['date']]=[]
        temp.append(str(sortdf.iloc[i]['title']).strip("\n"))
        """
        try:
            if('http' not in sortdf.iloc[i]['url'] ):
                try:
                    corrpt[sortdf.iloc[i]['date']].append(sortdf.iloc[i]['url'])
                    temp.append(str(sortdf.iloc[i]['url']).strip("\n"))
                except:
                    corrpt[sortdf.iloc[i]['date']]=[]
                    corrpt[sortdf.iloc[i]['date']].append(sortdf.iloc[i]['url'].rstrip("\n"))
                    temp.append(str(sortdf.iloc[i]['url']).strip("\n"))

                    
        #The data is already added before , so nothing to worry about.
        except Exception as e:
            print(e)
        """
        data[sortdf.iloc[i]['date']].append(' '.join(temp))
    
      for i in data:
        data[i]=set(data[i])
      st=[]
      mt=[]
      lt=[]
      mapping={"long":lt,"medium":mt,"short":st}
      uniq_dt=df2["trade_date"].unique()
      for i in range(len(uniq_dt)):
        
        #last but one since the next day is always for prediction.
        if(i<len(uniq_dt)-1):
            st.append(uniq_dt[i])
            if(i-7>=0):
                mt.append(uniq_dt[i-7:i])
            else:
                mt.append(uniq_dt[:i+1])
            if(i-30>=0):
                lt.append(uniq_dt[i-30:i])
            else:
                lt.append(uniq_dt[:i+1])
      print("Doing for "+stock)
      #print(mapping)
    
      #here term is for the time period, long, medium and short
      for term in mapping:
        count=0
        for dat in mapping[term]:
            temp=[]
            if(type(dat)==object):
              
                #for entry in dat:
                try:
                  temp.append(vectors[stock])
                except Exception as e:
                  pass
                print(len(temp))
                if(len(temp)!=0):
                    final_data[stock][term].append(np.array(temp))
                else:
                    final_data[stock][term].append(np.zeros(300))
            else:
                try:
                    final_data[stock][term].append(vectors[stock][dat])
                except Exception as e:
                    final_data[stock][term].append(np.zeros(300))
            count+=1
      for term in ['long','medium','short']:
        if(term=='long'):
            wall=30
        elif(term=='medium'):
            wall=7
        else:
            wall=1
        print("Resizing all vectors to same size for "  +stock+" "+term)
        for dat in range(len(final_data[stock][term])):
            ref=len(final_data[stock][term][dat])
            if(ref==300):
                final_data[stock][term][dat]=np.zeros((wall,300))
                continue
            else:
                if(wall-ref>0):
                    print(final_data[stock][term][dat].shape)
                    
                    final_data[stock][term][dat]=np.vstack((final_data[stock][term][dat],np.zeros((wall-ref,300))))
      print("Shapes after resize")
      for term in final_data[stock]:
        final_data[stock][term]=np.array(final_data[stock][term])
        print(" for "+term+"-"+str(final_data[stock][term].shape))
      print("Length of the change in price for "+stock,len(close_price))
      dt={"X":final_data[stock],"Y":np.array(close_price)}
      with open(".../Independent Project/Data/Vec training/20221030_100days/"+stock+"-lms_vec_training.pkl","wb")as f:
        pickle.dump(dt,f)
      print("done for "+stock)
      print("#-"*15)
print("Pre processing done !")
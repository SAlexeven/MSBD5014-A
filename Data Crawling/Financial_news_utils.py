# -*- coding: utf-8 -*-


from unicodedata import bidirectional
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup as bf

import re
import json
import torch
from bs4.element import CharsetMetaAttributeValue
from tqdm import tqdm
import time
import random
import tushare as ts
import ast
import datetime
import requests
import akshare as ak
import sys
from multiprocessing import Pool
import torch

"""-----------------------------------------------"""

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'}
cookies = {
    '__auc=e338969417ef7344ae449919c12; __asc=4cb7c69617f201cbce9bf466dca; __utma=177965731.1778915295.1644824906.1644824906.1645511229.2; __utmc=177965731; __utmz=177965731.1645511229.2.2.utmcsr=google|utmccn=(organic)|utmcmd=organic|utmctr=(not%20provided); CookiePolicyCheck=0; _cc_id=af14ad876611b8e47d20db0ad43c1052; __utma=81143559.738177882.1644824906.1644824906.1645511246.2; __utmc=81143559; __utmz=81143559.1645511246.2.2.utmcsr=aastocks.com|utmccn=(referral)|utmcmd=referral|utmcct=/; NewsZoomLevel=3; ASP.NET_SessionId=bt3zwgrxrakas3zczxxsv4bi; mLang=TC; AALive=3429; cto_bundle=rXZ9LV8lMkIwZElmOVNvWTFyd2lDNGdIM3FiaWtwSVBJcXVxMW4xeDk3SDNHdE9ZNzNLVCUyRnhJJTJGeUE5a2xkTklBTzRiWVBQUFhLT0g3R1RRYjBVanFVbVBwM0RlR2lNN0hFcU5KOFBLcHZ5SyUyRlZUR3dmaVMxOE0yZE1NNHZjMk5adXpPbEFSdHlqQ0M3b01sQnMxOWowZ3lHQXJqQSUzRCUzRA; __gads=ID=ee522e9dfd1d4c35-22c82040b6d0004c:T=1644824906:RT=1645515431:S=ALNI_MZWY7mcN-MhVR0tdkFGR1Uu3F4XYA; aa_cookie=218.102.252.238_62750_1645513384; __utmt_a3=1; __utmb=177965731.71.9.1645515545042; __utmt_a2=1; __utmt_b=1; __utmb=81143559.153.9.1645515500316'
}

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def returnN(x):
    return None if len(x) == 0 else x

def get_aastock_top_news():
    link = "http://www.aastocks.com/sc/stocks/news/aafn/top-news/0"
    response = requests.get(link, headers=headers)
    response.raise_for_status()  # raises exception when not a 2xx response
    if response.status_code != 204:
      return response.json()
    response.encoding = 'utf8'
    obj = bf(response.content, 'html.parser', from_encoding='gb18030')
    temp = obj.find_all('div',attrs={"class": "newshead4 mar2B lettersp2"})
    for i in temp:
        print(i.getText())
    temp = obj.find_all('div', attrs={"class": "newshead4 mar4B lettersp2"})
    for i in temp:
        print(i.getText())
    link = "http://www.aastocks.com/sc/resources/datafeed/getstockrecentnews.ashx?rc=10"
    response = requests.get(link, headers=headers)
    obj = ast.literal_eval(response.content.decode('utf-8'))
    print(obj)

def get_cctv_news():
    # crawl for recent 100 days cctv news
    today = datetime.datetime.today()
    news_list = []
    for i in tqdm(range(100)):
        day = today - datetime.timedelta(days=i)
        day = day.strftime('%Y%m%d')
        df = ak.news_cctv(date=day)
        df = df.to_dict('records')
        news_list.extend(df)
        time.sleep(1)
    file_location = open('/content/drive/MyDrive/Independent Project/Data/News data/cctv_news_100day.json','w', encoding="utf-8")
    json.dump(news_list, file_location, indent=1, ensure_ascii=False)
    print("Recent 100 days cctv news has been saved to 'data/cctv_news_100day.json'.")

def get_tvb_news():
    # crawl for recent 100 days tvb news
    today = datetime.datetime.today()
    news_list = []
    for i in tqdm(range(100)):
        day = today - datetime.timedelta(days=i)
        day = day.strftime('%Y%m%d')
        link = "https://api.news.tvb.com/news/v2.2.1/entry?category=finance&date="+day+"&profile=web"
        response = requests.get(link, headers=headers)
        """response.raise_for_status()  # raises exception when not a 2xx response
        if (response.status_code != 204 and response.headers["content-type"].strip().startswith("application/json")):
          try:
            return response.json()
          except ValueError:
            print("Unexpected error, please dismiss it.")"""
        
        #obj = json.loads(response.content)
        obj = response.json()
        target = obj['items']
        news_list.extend(target)
        time.sleep(1)
    #file_location = open('/content/drive/MyDrive/Independent Project/Data/News data/tvb_news_100day.json','w', encoding="utf-8")
    json_news_data=json.dumps(news_list, indent=1, ensure_ascii=False)
    with open('/content/drive/MyDrive/Independent Project/Data/News data/tvb_news_100day.json','w', encoding="utf-8") as f1:
      f1.write(json_news_data)
    print("Recent 100 days tvb news has been saved to 'News data/tvb_news_100day.json'.")

def get_tushare_news_thisyear(start_date_str):
    ts.set_token('2d7d464e46a43f8e58efcac32727985d0c31de986a87c2e5053d8761')
    pro = ts.pro_api()
    datetime_str = start_date_str + ' 00:00:00'
    today = datetime.datetime.strptime(datetime_str, '%Y/%m/%d %H:%M:%S')
    #today = datetime.datetime.today()
    
    for i in ['yuncaijing']:
        news_list = []
        for t in tqdm(range(0, 302)):
            end_day = (today - datetime.timedelta(days=t)).strftime('%Y-%m-%d')+' 23:59:59'
            start_day = (today - datetime.timedelta(days=t)).strftime('%Y-%m-%d')+' 00:00:00'
            df = pro.news(src=i, start_date=start_day, end_date=end_day)
            news_list.extend(df.to_dict("records"))
            time.sleep(100)
            if t ==99:
              file_location = open('.../Independent Project/Data/News data/'+i+'_news_yuncaijing_100days.json', 'w', encoding="utf-8")
              json.dump(news_list, file_location, indent=1, ensure_ascii=False)
            if t ==199:
              file_location = open('.../Independent Project/Data/News data/'+i+'_news_yuncaijing_200days.json', 'w', encoding="utf-8")
              json.dump(news_list, file_location, indent=1, ensure_ascii=False)
            if t ==299:
              file_location = open('.../Independent Project/Data/News data/'+i+'_news_yuncaijing_300days.json', 'w', encoding="utf-8")
              json.dump(news_list, file_location, indent=1, ensure_ascii=False)
            #end_day = (today - datetime.timedelta(days=t)).strftime('%Y-%m-%d')+' 23:59:59'
            #start_day = (today - datetime.timedelta(days=t)).strftime('%Y-%m-%d')+' 12:00:01'
            #df = pro.news(src=i, start_date=start_day, end_date=end_day)
            #news_list.extend(df.to_dict("records"))
            #time.sleep(60)
        for news in news_list:
            news['source'] = i
        file_location = open('/content/drive/MyDrive/Independent Project/Data/News data/'+i+'_news_yuncaijing_thisyear_to20221030.json', 'w', encoding="utf-8")
        json.dump(news_list, file_location, indent=1, ensure_ascii=False)
        time.sleep(120)
    print("Recent days news from " + start_date_str + "has been saved.")

def get_tushare_news_100days(start_date_str):
    ts.set_token('2d7d464e46a43f8e58efcac32727985d0c31de986a87c2e5053d8761')
    pro = ts.pro_api()
    datetime_str = start_date_str + ' 00:00:00'
    today = datetime.datetime.strptime(datetime_str, '%Y/%m/%d %H:%M:%S')
    #today = datetime.datetime.today()
    
    for i in ['yuncaijing']:
        news_list = []
        for t in tqdm(range(0, 100)):
            end_day = (today - datetime.timedelta(days=t)).strftime('%Y-%m-%d')+' 23:59:59'
            start_day = (today - datetime.timedelta(days=t)).strftime('%Y-%m-%d')+' 00:00:00'
            df = pro.news(src=i, start_date=start_day, end_date=end_day)
            news_list.extend(df.to_dict("records"))
            time.sleep(100)
            #end_day = (today - datetime.timedelta(days=t)).strftime('%Y-%m-%d')+' 23:59:59'
            #start_day = (today - datetime.timedelta(days=t)).strftime('%Y-%m-%d')+' 12:00:01'
            #df = pro.news(src=i, start_date=start_day, end_date=end_day)
            #news_list.extend(df.to_dict("records"))
            #time.sleep(60)
        for news in news_list:
            news['source'] = i
        file_location = open('/content/drive/MyDrive/Independent Project/Data/News data/'+i+'_news_yuncaijing_100days_to20221030.json', 'w', encoding="utf-8")
        json.dump(news_list, file_location, indent=1, ensure_ascii=False)
        time.sleep(120)
    print("Recent days news from " + start_date_str + "has been saved.")

def get_tushare_news_1year(start_date_str):
    ts.set_token('2d7d464e46a43f8e58efcac32727985d0c31de986a87c2e5053d8761')
    pro = ts.pro_api()
    datetime_str = start_date_str + ' 00:00:00'
    today = datetime.datetime.strptime(datetime_str, '%Y/%m/%d %H:%M:%S')
    for tt in range(0,1):
        print("start a new year")
        for i in ['10jqka', 'eastmoney', 'yuncaijing']:
            news_list = []
            for t in tqdm(range(0, 365)):
                end_day = (today - datetime.timedelta(days=t)).strftime('%Y-%m-%d')+' 12:00:00'
                start_day = (today - datetime.timedelta(days=t)).strftime('%Y-%m-%d')+' 00:00:00'
                df = pro.news(src=i, start_date=start_day, end_date=end_day)
                news_list.extend(df.to_dict("records"))
                time.sleep(60)
                end_day = (today - datetime.timedelta(days=t)).strftime('%Y-%m-%d')+' 23:59:59'
                start_day = (today - datetime.timedelta(days=t)).strftime('%Y-%m-%d')+' 12:00:01'
                df = pro.news(src=i, start_date=start_day, end_date=end_day)
                news_list.extend(df.to_dict("records"))
                time.sleep(60)
            for news in news_list:
                news['source'] = i
            file_location = open('/content/drive/MyDrive/Independent Project/Data/News data/'+i+'_news_'+today.strftime('%Y%m%d')+'.json', 'w', encoding="utf-8")
            json.dump(news_list, file_location, indent=1, ensure_ascii=False)
            time.sleep(100)
        today = today - datetime.timedelta(days=365)
    print("Recent 1 year news from " + start_date_str + "has been saved.")

if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  pool=Pool(processes=10)
  pool.map(get_tushare_news_100days("2022/10/30").to(device))
  #get_tushare_news_thisyear("2022/10/30")
  #get_tvb_news()



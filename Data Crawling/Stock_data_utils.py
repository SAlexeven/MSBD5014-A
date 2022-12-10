# -*- coding: utf-8 -*-

import json
import sys
import tushare as ts
import pandas as pd
import os
import yfinance as yf
import akshare as ak
from datetime import datetime
import re

# from influxdb_client import InfluxDBClient, Point, WritePrecision
# from influxdb_client.client.write_api import SYNCHRONOUS

"""You can generate an API token from the "API Tokens Tab" in the UI"""

token = "tvS_7TrFovJtUBQFCb1l9vE03K8MEkiq0tKzdQ_bATmxztfRqyNUoddyZUCVLf-ilGB9Oi8xyCYjuOUQO_HHmg=="
org = "hkust"
bucket = "hkust"

class TS:
    def __init__(self, key):
        self.pro = ts.pro_api(key)
    def get_change(self, stock_code_list, start_date, end_date):
        res = []
        series_len = None
        for stock_code in stock_code_list:
            x = self.pro.daily(ts_code=stock_code, start_date=start_date, end_date=end_date)[:]['change'].tolist()
            res.append({
                'stock_code': stock_code,
                'hist_change': x
            })
            if series_len is not None:
                if series_len > len(x):
                    series_len = len(x)
            else:
                series_len = len(x)
        for i in res:
            i['hist_change'] = i['hist_change'][:series_len]
        return res
    def get_close(self, stock_code_list, start_date, end_date):
        res = []
        series_len = None
        for stock_code in stock_code_list:
            x = self.pro.daily(ts_code=stock_code, start_date=start_date, end_date=end_date)[:]['close'].tolist()
            res.append({
                'stock_code': stock_code,
                'hist_close': x
            })
            if series_len is not None:
                if series_len > len(x):
                    series_len = len(x)
            else:
                series_len = len(x)
        for i in res:
            i['hist_close'] = i['hist_close'][:series_len]
        return res

def deal_with_ts_code(s, mode='only_num'):
    if mode == 'only_num':
        return re.sub("\D", "", s)
    if mode == 'only_character':
        return re.sub(u"([^\u0041-\u005a\u0061-\u007a])", "", s)
    if mode == 'hk':
        # 需要去掉第一个0在最后加上.HK
        return str(s[1:]+'.HK')
    if mode == 'hs_symbol':
        if 'SH' in s:
            return re.sub("\D", "", s) + '.SS'
        else:
            return re.sub("\D", "", s) + '.SZ'

class DataLoader:
    """
    目前dataloader仅支持使用沪深股票数据的其中1000支股票,
    就是data/stock_index.jon 中的前1000支。
    测试阶段仅包含20180101到20191231每天的数据，
    存储为pkl格式节省空间
    """
    def __init__(self):
        #index_location = os.path.abspath(os.path.join(os.getcwd(), "../", "data"))
        index_location = ".../Independent Project/Data/Index location"
        stockdata_location = ".../Independent Project/Data/Stock data"
        index_file = os.path.join(index_location, 'stock_index.json')
        self.us_index_file = os.path.join(index_location, 'US_stock_index.json')
        self.hk_index_file = os.path.join(index_location, 'HK_stock_index.json')
        self.hk_index_excel = os.path.join(index_location, 'Hang Seng China (Hong Kong-listed) 25 Index.xlsx')
        self.stockdata_location = stockdata_location
        self.get_year_string(2022,2022)
        #self.stock_list = json.load(open(index_file))


    @staticmethod
    def get_year_string(start_year, end_year):
        res = []
        will_stop_year = start_year
        if start_year % 2 == 0:
            res.append(str(will_stop_year) + "_" + str(will_stop_year + 1))
            while will_stop_year + 1 < end_year:
                will_stop_year += 2
                res.append(str(will_stop_year) + "_" + str(will_stop_year + 1))
        else:
            res.append(str(will_stop_year - 1) + "_" + str(will_stop_year))
            while will_stop_year < end_year:
                will_stop_year += 2
                res.append(str(will_stop_year - 1) + "_" + str(will_stop_year))
        return res


    def find_stock_tushare(self, ts_code, start_date=20180101, end_date=20191231, tuning_mode=False):
        """
        使用find_stock func输入ts_code查询股票
        :param tuning_mode: use to turn on tuning mode
        :param start_date: 查询开始日期
        :param end_date: 查询结束日期
        :param ts_code: 股票代码
        :return: dataframe
        """
        try:
            index = self.stock_list.index(ts_code)
            if tuning_mode:
                print(index)
        except:
            print(ts_code + 'is not in local stock list, try to connect to eastmoney to get data')
            stock_data = ak.stock_zh_a_hist(symbol=deal_with_ts_code(ts_code),
                                            period="daily", start_date=str(start_date),
                                            end_date=str(end_date), adjust="")
            stock_data[['ts_code']] = ts_code
            stock_data = stock_data.rename(columns={
                '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '日期': 'trade_date',
                '涨跌额': 'change', '涨跌幅': 'pct_chg', '成交量': 'vol', '成交额': 'amount'
            })
            stock_data['trade_date'] = stock_data['trade_date'].apply(lambda x: int(''.join(x.split("-"))))
            return stock_data
        if index is not None:
            if start_date > end_date:
                print("check your start date and end date.")
                return None
            start_year, end_year = start_date // 10000, end_date // 10000
            folder_group = self.get_year_string(start_year, end_year)
            df_list = []
            for folder in folder_group:
                try:
                    data_location = os.path.abspath(os.path.join(os.getcwd(), "..", "data", folder))
                    file_name = os.path.join(data_location, str(index // 1000) + '_HS.pkl')
                    df = pd.read_pickle(file_name)
                except:
                    print("local record is not updated, connect to eastmoney data source.")
                    stock_data = ak.stock_zh_a_hist(symbol=deal_with_ts_code(ts_code),
                                                    period="daily", start_date=str(start_date),
                                                    end_date=str(end_date), adjust="")
                    stock_data[['ts_code']] = ts_code
                    stock_data = stock_data.rename(columns={
                        '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '日期': 'trade_date',
                        '涨跌额': 'change', '涨跌幅': 'pct_chg', '成交量': 'vol', '成交额': 'amount'
                    })
                    stock_data['trade_date'] = stock_data['trade_date'].apply(lambda x: int(''.join(x.split("-"))))
                    return stock_data
                df = df.loc[(df['ts_code'] == ts_code) & (df['trade_date'] >= str(start_date))
                            & (df['trade_date'] <= str(end_date))]
                df_list.append(df)
            result = pd.concat(df_list)
            if result.shape[0] == 0:
                print(ts_code)
                print("no local record founded, connect to eastmoney data source.")
                stock_data = ak.stock_zh_a_hist(symbol=deal_with_ts_code(ts_code),
                                                period="daily", start_date=str(start_date),
                                                end_date=str(end_date), adjust="")
                stock_data[['ts_code']] = ts_code
                stock_data = stock_data.rename(columns={
                    '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '日期': 'trade_date',
                    '涨跌额': 'change', '涨跌幅': 'pct_chg', '成交量': 'vol', '成交额': 'amount'
                })
                stock_data['trade_date'] = stock_data['trade_date'].apply(lambda x: int(''.join(x.split("-"))))
                return stock_data
            else:
                return pd.concat(df_list)
        else:
            return None
    def find_stock(self, ts_code, start_date=20180101, end_date=20191231, tuning_mode=False):
        """
        使用find_stock func输入ts_code查询股票，和find_stock_test的区别是使用本地的东财数据
        :param start_date: 查询开始日期
        :param end_date: 查询结束日期
        :param ts_code: 股票代码
        :return: dataframe
        """
        try:
            index = self.stock_list.index(ts_code)
            if tuning_mode:
                print(index)
        except:
            print(ts_code + 'is not in local stock list, try to connect to eastmoney to get data')
            stock_data = ak.stock_zh_a_hist(symbol=deal_with_ts_code(ts_code),
                                            period="daily", start_date=str(start_date),
                                            end_date=str(end_date), adjust="")
            stock_data[['ts_code']] = ts_code
            stock_data = stock_data.rename(columns={
                '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '日期': 'trade_date',
                '涨跌额': 'change', '涨跌幅': 'pct_chg', '成交量': 'vol', '成交额': 'amount'
            })
            stock_data['trade_date'] = stock_data['trade_date'].apply(lambda x: int(''.join(x.split("-"))))
            return stock_data
        if index is not None:
            if start_date > end_date:
                print("check your start date and end date.")
                return None
            start_year, end_year = start_date // 10000, end_date // 10000
            folder_group = self.get_year_string(start_year, end_year)
            df_list = []
            for folder in folder_group:
                try:
                    data_location = os.path.abspath(os.path.join(os.getcwd(), "..", "data", folder))
                    file_name = os.path.join(data_location, str(index // 1000) + '_HS_em.pkl')
                    df = pd.read_pickle(file_name)
                except:
                    print("local record is not updated, connect to eastmoney data source.")
                    stock_data = ak.stock_zh_a_hist(symbol=deal_with_ts_code(ts_code),
                                                    period="daily", start_date=str(start_date),
                                                    end_date=str(end_date), adjust="")
                    stock_data[['ts_code']] = ts_code
                    stock_data = stock_data.rename(columns={
                        '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '日期': 'trade_date',
                        '涨跌额': 'change', '涨跌幅': 'pct_chg', '成交量': 'vol', '成交额': 'amount'
                    })
                    stock_data['trade_date'] = stock_data['trade_date'].apply(lambda x: int(''.join(x.split("-"))))
                    return stock_data
                df = df.loc[(df['ts_code'] == ts_code) & (df['trade_date'] >= start_date)
                            & (df['trade_date'] <= end_date)]
                if tuning_mode:
                    print(df)
                df = df.rename(columns={'成交量': 'vol', '成交额': 'amount'})
                df_list.append(df)
            result = pd.concat(df_list)
            if result.shape[0] == 0:
                print(ts_code)
                print("no local record founded, connect to eastmoney data source.")
                return None
                # stock_data = ak.stock_zh_a_hist(symbol=deal_with_ts_code(ts_code),
                #                                 period="daily", start_date=str(start_date),
                #                                 end_date=str(end_date), adjust="")
                # stock_data[['ts_code']] = ts_code
                # stock_data = stock_data.rename(columns={
                #     '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '日期': 'trade_date',
                #     '涨跌额': 'change', '涨跌幅': 'pct_chg', '成交量': 'vol', '成交额': 'amount'
                # })
                # stock_data['trade_date'] = stock_data['trade_date'].apply(lambda x: int(''.join(x.split("-"))))
                # return stock_data
            else:
                return pd.concat(df_list)
        else:
            return None
    def find_us_stock(self, ts_code, start_date=20180101, end_date=20191231):
        """
        使用find_stock func输入ts_code查询股票
        :param start_date: 查询开始日期
        :param end_date: 查询结束日期
        :param ts_code: 股票代码
        :return: dataframe 美股的后复权数据
        """
        us_stock_list = json.load(open(self.us_index_file))
        try:
            index = us_stock_list.index(ts_code)
        except:
            print(ts_code + 'is not in local stock list, try to connect to eastmoney to get data.')
            stock_data = ak.stock_us_hist(symbol=deal_with_ts_code(ts_code), start_date=str(start_date),
                                          end_date=str(end_date),
                                          adjust="hfq")
            stock_data[['ts_code']] = ts_code
            stock_data = stock_data.rename(columns={
                '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '日期': 'trade_date',
                '涨跌额': 'change', '涨跌幅': 'pct_chg', '成交量': 'vol', '成交额': 'amount'
            })
            stock_data['trade_date'] = stock_data['trade_date'].apply(lambda x: int(''.join(x.split("-"))))
            return stock_data
        if index is not None:
            if start_date > end_date:
                print('start query date is larger than end query date')
                return None
            start_year, end_year = start_date // 10000, end_date // 10000
            folder_group = self.get_year_string(start_year, end_year)
            df_list = []
            for folder in folder_group:
                try:
                    data_location = os.path.abspath(os.path.join(os.getcwd(), "..", "data", folder))
                    file_name = os.path.join(data_location, str(index // 1000) + '_US_hfq.pkl')
                    df = pd.read_pickle(file_name)
                except:
                    print("local record is not updated, connect to eastmoney data source.")
                    stock_data = ak.stock_us_hist(symbol=deal_with_ts_code(ts_code), start_date=str(start_date),
                                                  end_date=str(end_date),
                                                  adjust="hfq")
                    stock_data[['ts_code']] = ts_code
                    stock_data = stock_data.rename(columns={
                        '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '日期': 'trade_date',
                        '涨跌额': 'change', '涨跌幅': 'pct_chg', '成交量': 'vol', '成交额': 'amount'
                    })
                    stock_data['trade_date'] = stock_data['trade_date'].apply(lambda x: int(''.join(x.split("-"))))
                    return stock_data
                df['trade_date'] = df['日期'].apply(lambda xx: int("".join(xx.split("-"))))
                df = df.loc[(df['code'] == ts_code) & (df['trade_date'] >= start_date)
                            & (df['trade_date'] <= end_date)]
                df = df.rename(columns={
                    '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low',
                    '涨跌额': 'change', '涨跌幅': 'pct_chg', 'code': 'ts_code', '成交量': 'vol', '成交额': 'amount'
                })
                df_list.append(df)
            result = pd.concat(df_list)
            if result.shape[0] == 0:
                print(ts_code)
                print("no local record founded, connect to eastmoney data source.")
                stock_data = ak.stock_us_hist(symbol=deal_with_ts_code(ts_code), start_date=str(start_date),
                                              end_date=str(end_date),
                                              adjust="hfq")
                stock_data[['ts_code']] = ts_code
                stock_data = stock_data.rename(columns={
                    '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '日期': 'trade_date',
                    '涨跌额': 'change', '涨跌幅': 'pct_chg', '成交量': 'vol', '成交额': 'amount'
                })
                stock_data['trade_date'] = stock_data['trade_date'].apply(lambda x: int(''.join(x.split("-"))))
                return stock_data
            else:
                return result
        else:
            return None
    def find_hk_stock(self, hk_index_excel, stockdata_location, start_date=20220101, end_date=20221030):


        """
        使用find_stock func输入ts_code查询股票
        :param start_date: 查询开始日期
        :param end_date: 查询结束日期
        :param ts_code: 股票代码
        :return: dataframe 无复权的港股数据
        """

        #hk_stock_dict = json.load(open(self.hk_index_file))
        #hk_stock_list = list(hk_stock_dict.values())
        hk_index_excel = self.hk_index_excel
    
        hk_stock_df = pd.read_excel(self.hk_index_excel,header=0,converters={'ts_code':str})

        for i in hk_stock_df['ts_code']:
          ts_code = i
          print(ts_code,'is not in local stock list, try to connect to eastmoney to get data.')
         
          stock_data = ak.stock_hk_hist(symbol=ts_code, start_date=str(start_date),
                                          end_date=str(end_date),adjust="")
        
          stock_data[['ts_code']] = ts_code
          stock_data = stock_data.rename(columns={
                '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '日期': 'trade_date',
                '涨跌额': 'change', '涨跌幅': 'pct_chg', '成交量': 'vol', '成交额': 'amount'
          })
          stockdata_filepath = self.stockdata_location + "/" + "20120101-20221030"
          if not os.path.exists(stockdata_filepath):
            os.makedirs(stockdata_filepath)
          #stockdata_filepath_i = os.path.join(self.stockdata_location, '20220101-20221030.csv') 
          if stock_data.shape[0] > 0:
              stock_data['trade_date'] = stock_data['trade_date'].apply(lambda x: int(''.join(x.split("-"))))
          stock_data.to_csv(stockdata_filepath + "/" + ts_code +'.csv') 
          #return stock_data
              
             

@staticmethod
    def get_year_string(start_year, end_year):
        res = []
        will_stop_year = start_year
        if start_year % 2 == 0:
            res.append(str(will_stop_year) + "_" + str(will_stop_year + 1))
            while will_stop_year + 1 < end_year:
                will_stop_year += 2
                res.append(str(will_stop_year) + "_" + str(will_stop_year + 1))
        else:
            res.append(str(will_stop_year - 1) + "_" + str(will_stop_year))
            while will_stop_year < end_year:
                will_stop_year += 2
                res.append(str(will_stop_year - 1) + "_" + str(will_stop_year))
        return res

if __name__ == '__main__':
    """
    想要直接测试tushare需要在官网进行注册拿到token
    低频数据获取：
    美股，港股通过yahoo finance api获得
    A股通过tushare获得，akshare没有进行处理数据质量不高
    高频数据获取:
    美股，港股可以通过yahoo finance api获得，1m数据仅存储30天，
    其他频率小于一天的数据仅存储60天
    A股可以通过akshare获取，1m数据存储4个月
    influx db test:
    token = tvS_7TrFovJtUBQFCb1l9vE03K8MEkiq0tKzdQ_bATmxztfRqyNUoddyZUCVLf-ilGB9Oi8xyCYjuOUQO_HHmg==
    """
    # x = yf.Ticker("GOOGl")
    # print(type(x.history(interval='1d', start='2020-12-01', end='2020-12-10')))

    # x = ak.stock_zh_a_minute(symbol='sh600751', period='1', adjust='qfq')
    # 测试dataloader

    x = DataLoader()
    # x.find_stock('600805.SH', start_date=20180610, end_date=20190610) 
    hk_index_excel =".../Independent Project/Data/Index location/Hang Seng China (Hong Kong-listed) 25 Index.xlsx"
    stockdata_location = ".../Independent Project/Data/Stock data"
    print(x.find_hk_stock(hk_index_excel, stockdata_location, start_date=20220101, end_date=20221030))
    # index_location = os.path.abspath(os.path.join(os.getcwd(), "..", "data"))
    # index_file = os.path.join(index_location, 'HS_stock_index.json')
    # stock_list = json.load(open(index_file))
    # print(x.find_stock_test(stock_list[1234], start_date=20211201, end_date=20211230))
    # -------------------
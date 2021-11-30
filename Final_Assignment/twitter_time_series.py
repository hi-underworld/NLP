# Chap02-03/twitter_time_series.py
import sys
import json
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import pandas as pd
import numpy as np
import pickle

if __name__ == '__main__':

    fname = 'data/TweetsNBA_1000.json'

    with open(fname, 'r') as f:
        all_dates = []
        for line in f:
            tweet = json.loads(line)
            all_dates.append(tweet.get('created_at'))
        ones = np.ones(len(all_dates))
        idx = pd.DatetimeIndex(all_dates)
        # the actual series (at series of 1s for the moment)
        my_series = pd.Series(ones, index=idx)

        # Resampling / bucketing into 1-minute buckets
        # per_minute = my_series.resample('1Min', how='sum').fillna(0)
        per_minute = my_series.resample('1Min').sum().fillna(0)
        print(my_series.head())
        print(per_minute.head())
        
        fig, ax = plt.subplots()
        ax.grid(True)
        ax.set_title("Tweet Frequencies")
        
        hours = mdates.MinuteLocator(interval=1)
        date_formatter = mdates.DateFormatter('%H:%M')

        datemin = datetime(2018, 6, 7, 1, 12)   # 设置date坐标轴的最小值
        datemax = datetime(2018, 6, 7, 1, 16)   # 设置date坐标轴的最小值

        ax.xaxis.set_major_locator(hours)       # 将间隔调整为hours
        ax.xaxis.set_major_formatter(date_formatter)
                                                # 使用date_formatter时间
        ax.set_xlim(datemin, datemax)           # 设置x轴的范围
        max_freq = per_minute.max()             # 将per_minute的最大值
        ax.set_ylim(0, max_freq)                # 设置为y轴可选的最大范围
        ax.plot(per_minute.index, per_minute)   # 绘制图像

        plt.savefig('tweet_time_series.png')    # 将图像进行保存

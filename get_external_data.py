#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
获取外部数据脚本：
1. 从 kl8_order_data.csv 中读取开奖日期的开始日期（数据集中的最后一天仅用于展示范围，但不作为截止日期）。
2. 使用 Meteostat 获取对应日期内北京天气数据：
   - Daily 获取平均温度（tavg）和平均风速（wspd）
   - Hourly 获取相对湿度（rhum），聚合为每日平均湿度
3. 使用 pytrends 获取“彩票”关键词在中国的搜索热度数据（失败时自动重试，不使用代理若均失败则生成模拟数据）。
4. 合并数据并输出 CSV 文件 external_data.csv 。

依赖库：
    pip install meteostat pytrends pandas
"""

import os
import random
import pandas as pd
from datetime import datetime

# 使用 Meteostat 获取天气数据
from meteostat import Point, Daily, Hourly

# 尝试导入 pytrends
try:
    from pytrends.request import TrendReq
    HAS_PYTREND = True
except ImportError:
    HAS_PYTREND = False


def get_date_range(data_path):
    """
    读取彩票数据，返回开奖日期最早日期。
    数据集中的最后一天仅用作展示范围，不作为后续数据获取的截止日期。
    """
    print("正在获取数据的日期范围...")
    df = pd.read_csv(data_path, encoding='utf-8', parse_dates=["开奖日期"])
    start_date = df["开奖日期"].min().date()
    file_end_date = df["开奖日期"].max().date()
    print(f"数据集中的日期范围：{start_date} 到 {file_end_date}")
    return start_date, file_end_date


def get_weather_data(start_date, end_date):
    """
    获取北京对应日期内的每日天气数据，包含：
      - 平均温度 (tavg) 来自 Daily 数据
      - 平均风速 (wspd) 来自 Daily 数据
      - 相对湿度 (rhum) 来自 Hourly 数据，聚合为每日平均值
    """
    print("正在获取北京天气数据...")
    # 北京坐标
    beijing = Point(39.9042, 116.4074)
    start = datetime.combine(start_date, datetime.min.time())
    end = datetime.combine(end_date, datetime.min.time())

    # 获取每日数据（温度与风速）
    daily_data = Daily(beijing, start, end)
    daily_data = daily_data.fetch()
    daily_data.reset_index(inplace=True)

    # 获取每小时数据（湿度）—— 注意字段名称为 'rhum'
    hourly_data = Hourly(beijing, start, end)
    hourly_data = hourly_data.fetch()
    if not hourly_data.empty and 'rhum' in hourly_data.columns:
        hourly_data = hourly_data.reset_index()
        hourly_data['Date'] = hourly_data['time'].dt.date
        # 计算每日平均湿度
        daily_rhum = hourly_data.groupby('Date')['rhum'].mean().reset_index()
    else:
        print("未获取到 hourly 'rhum' 数据，湿度将设置为 NaN")
        daily_rhum = pd.DataFrame({'Date': daily_data['time'].dt.date, 'rhum': float('nan')})

    # 处理 Daily 数据：提取日期、平均温度与平均风速
    weather = daily_data[['time', 'tavg', 'wspd']].rename(
        columns={'time': 'Date', 'tavg': 'temperature', 'wspd': 'wind_speed'}
    )
    weather['Date'] = weather['Date'].dt.date

    # 合并温度、风速和湿度数据
    weather = pd.merge(weather, daily_rhum, on='Date', how='left')
    # 使用前向填充避免缺失值
    weather['temperature'] = weather['temperature'].ffill()
    weather['wind_speed'] = weather['wind_speed'].ffill()
    weather['rhum'] = weather['rhum'].ffill()
    # 重命名 'rhum' 为 'humidity'
    weather = weather.rename(columns={'rhum': 'humidity'})

    print("天气数据获取完成。")
    return weather


def get_search_trend_data(start_date, end_date):
    """
    获取“彩票”关键词在中国的搜索热度数据。

    尝试以下两种方案：
      1. 使用预设代理获取数据；
      2. 不使用代理获取数据。
    若均失败，则生成模拟数据返回。
    返回 DataFrame 包含 Date 和 search_trend 列。
    """
    print("正在获取搜索热度数据...")
    if not HAS_PYTREND:
        print("pytrends 库不可用，直接生成模拟数据。")

    # 设置完整的自定义 User-Agent 字符串以及 Accept-Language，以模拟常见浏览器请求
    custom_user_agent = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/107.0.5304.87 Safari/537.36"
    )

    # 定义两个代理设置：预设代理和无代理
    proxies_options = [
        {"http": "http://127.0.0.1:33331/commands/pac", "https": "http://127.0.0.1:33331/commands/pac"},
        None
    ]

    # 延长超时时间、设置重试次数和退避因子
    timeout_setting = (60, 120)  # (连接超时, 读取超时)
    retries_setting = 3
    backoff_factor_setting = 0.1

    timeframe = f"{start_date} {end_date}"
    kw_list = ["彩票"]

    for proxy in proxies_options:
        try:
            if proxy:
                pytrends = TrendReq(
                    hl='zh-CN',
                    tz=480,
                    proxies=proxy,
                    timeout=timeout_setting,
                    retries=retries_setting,
                    backoff_factor=backoff_factor_setting
                )
            else:
                pytrends = TrendReq(
                    hl='zh-CN',
                    tz=480,
                    timeout=timeout_setting,
                    retries=retries_setting,
                    backoff_factor=backoff_factor_setting
                )
            # 更新请求头，模拟正常浏览器请求
            pytrends.headers.update({
                'User-Agent': custom_user_agent,
                'Accept-Language': 'zh-CN,zh;q=0.9'
            })
            pytrends.build_payload(kw_list, timeframe=timeframe, geo='CN')
            data = pytrends.interest_over_time()
            if data.empty:
                raise ValueError("无搜索热度数据返回")
            trend = data.reset_index()[['date', '彩票']].rename(
                columns={'date': 'Date', '彩票': 'search_trend'}
            )
            trend['Date'] = trend['Date'].dt.date
            print(f"搜索热度数据获取完成，使用代理设置：{proxy if proxy else '无代理'}")
            return trend
        except Exception as e:
            print(f"尝试使用代理 {proxy if proxy else '无代理'} 获取搜索热度数据时出错：{e}")

    # 如果以上方式均失败，则生成模拟数据
    print("使用模拟搜索热度数据。")
    dates = pd.date_range(start=start_date, end=end_date).date
    trend = pd.DataFrame({
        'Date': dates,
        'search_trend': [random.randint(20, 100) for _ in range(len(dates))]
    })
    print("模拟搜索热度数据生成完成。")
    return trend


def merge_external_data(weather, trend):
    """按 Date 合并天气数据与搜索热度数据"""
    print("正在合并天气数据与搜索热度数据...")
    external = pd.merge(weather, trend, on='Date', how='left')
    external['search_trend'] = external['search_trend'].fillna(external['search_trend'].mean())
    print("数据合并完成。")
    return external


def main():
    # 设置 kl8_order_data.csv 路径（请根据实际路径调整）
    data_path = "/home/luolu/PycharmProjects/NeuralForecast/Utils/GetData/kl8/kl8_order_data.csv"

    print("=== 开始获取外部数据 ===")
    # 获取数据集中的开始日期（最后一天仅做提示，不作为截止日期）
    start_date, _ = get_date_range(data_path)
    # 截止日期设为当前本地（北京时间）的日期
    end_date = datetime.now().date()
    print(f"使用的截止日期：{end_date}")

    weather = get_weather_data(start_date, end_date)
    trend = get_search_trend_data(start_date, end_date)

    external = merge_external_data(weather, trend)

    # 将生成的外部数据保存到 external_data.csv，与 kl8_order_data.csv 同目录
    output_path = os.path.join(os.path.dirname(data_path), "external_data.csv")
    external.to_csv(output_path, index=False, encoding='utf-8')
    print(f"外部数据已保存至 {output_path}")
    print("=== 外部数据获取完毕 ===")


if __name__ == "__main__":
    main()

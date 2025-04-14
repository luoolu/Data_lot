#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 10/20/24
# @Author  : luoolu
# @Github  : https://luoolu.github.io
# @Software: PyCharm
# @File    : data.py
# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
彩票数据下载器（大师级别完整代码）

本脚本用于从乐彩网数据源下载三种彩票（超级大乐透、快乐8、双色球）的历史数据，
并支持两种数据版本：
  - 排好序（升序排列）数据
  - 出球顺序（原始开奖顺序）数据 —— 默认下载此版本

各彩票数据下载地址（文本格式）：
  超级大乐透：
    排序版（升序）： http://data.17500.cn/dlt2_asc.txt
    出球顺序版：      http://data.17500.cn/dlt_asc.txt
  快乐8：
    排序版（升序）： http://data.17500.cn/kl8_asc.txt
    出球顺序版：      http://data.17500.cn/kl82_asc.txt
  双色球：
    排序版（升序）： http://data.17500.cn/ssq_asc.txt
    出球顺序版：      http://data.17500.cn/ssq_asc.txt

【快乐8说明】
快乐8数据文件共 128 列，具体表头如下：
  1. 开奖期号
  2. 开奖日期
  3-22. 20个球的排好序的数字（依次命名为 "排好序_1", ..., "排好序_20"）
  23-42. 20个球的出球顺序（依次命名为 "出球顺序_1", ..., "出球顺序_20"）
  43-128. 各项投注统计数据（依次为：
       本期销售金额, 选十玩法奖池金额,
       选十中十注数, 单注奖金_十中十,
       选十中九注数, 单注奖金_十中九,
       选十中八注数, 单注奖金_十中八,
       选十中七注数, 单注奖金_十中七,
       选十中六注数, 单注奖金_十中六,
       选十中五注数, 单注奖金_十中五,
       选十中零注数, 单注奖金_十中零,
       选九中九注数, 单注奖金_九中九,
       选九中八注数, 单注奖金_九中八,
       选九中七注数, 单注奖金_九中七,
       选九中六注数, 单注奖金_九中六,
       选九中五注数, 单注奖金_九中五,
       选九中四注数, 单注奖金_九中四,
       选九中零注数, 单注奖金_九中零,
       选八中八注数, 单注金额_八中八,
       选八中七注数, 单注金额_八中七,
       选八中六注数, 单注金额_八中六,
       选八中五注数, 单注金额_八中五,
       选八中四注数, 单注金额_八中四,
       选八中零注数, 单注金额_八中零,
       选七中七注数, 单注金额_七中七,
       选七中六注数, 单注金额_七中六,
       选七中五注数, 单注金额_七中五,
       选七中四注数, 单注金额_七中四,
       选七中零注数, 单注金额_七中零,
       选六中六注数, 单注金额_六中六,
       选六中五注数, 单注金额_六中五,
       选六中四注数, 单注金额_六中四,
       选六中三注数, 单注金额_六中三,
       选五中五注数, 单注金额_五中五,
       选五中四注数, 单注金额_五中四,
       选五中三注数, 单注金额_五中三,
       四选中四注数, 单注金额_四中四,
       四选中三注数, 单注金额_四中三,
       四选中二注数, 单注金额_四中二,
       选三中三注数, 单注金额_三中三,
       选三中二注数, 单注金额_三中二,
       选二中二注数, 单注金额_二中二,
       选一中一注数, 单注金额_一中一 ]

运行示例：
  默认下载出球顺序数据：
      $ python lottery_downloader.py
  指定下载排好序数据：
      $ python lottery_downloader.py --data_type asc
"""

import os
import argparse
import pandas as pd
import requests
from io import StringIO


def ensure_directories():
    """确保存放数据的目录存在"""
    for folder in ['dlt', 'kl8', 'ssq']:
        os.makedirs(folder, exist_ok=True)


def download_and_save(url, columns, save_path, encoding='utf-8'):
    """
    下载数据、解析为 DataFrame 并保存到 CSV 文件
    :param url: 数据 URL
    :param columns: 指定的列名列表
    :param save_path: 保存 CSV 文件的路径
    :param encoding: 文件编码（默认 utf-8）
    :return: 解析后的 DataFrame 或 None（下载失败时）
    """
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            file_content = response.content.decode(encoding)
            df = pd.read_csv(StringIO(file_content), sep=r'\s+', header=None, names=columns, engine='python')
            df.to_csv(save_path, index=False, encoding=encoding)
            print(f"数据已保存到 {save_path}")
            return df
        else:
            print(f"下载失败。URL: {url}，HTTP 状态码: {response.status_code}")
    except Exception as e:
        print(f"下载或解析数据时出错: {e}")
    return None


# ===================== 超级大乐透 =====================
def get_dlt_asc_data():
    """
    下载大乐透排好序数据（升序排列版）
    来源文件：dlt2_asc.txt
    """
    url = 'http://data.17500.cn/dlt2_asc.txt'
    columns = [
        '开奖期号', 'Date',
        'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7',
        'sd1', 'sd2', 'sd3', 'sd4', 'sd5', 'sd6', 'sd7',
        "投注总额", "滚入下期一等奖池", "一等注数", "单注奖金_一等",
        "二等注数", "单注奖金_二等", "三等注数", "单注奖金_三等",
        "四等注数", "单注奖金_四等", "五等注数", "单注奖金_五等",
        "六等注数", "单注奖金_六等", "七等注数", "单注奖金_七等",
        "八等注数", "单注奖金_八等", "追加一等注数", "单注奖金_追加一等",
        "追加二等注数", "单注奖金_追加二等", "追加三等注数", "单注奖金_追加三等",
        '附加玩法投注总额', '一等奖注数', "单注奖金_附加", "r1", "r2"
    ]
    save_path = os.path.join('dlt', 'dlt_asc_data.csv')
    df = download_and_save(url, columns, save_path)
    if df is not None:
        latest_date = df['Date'].iloc[-1]
        latest_columns = ['Date', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7']
        latest_file = os.path.join('dlt', f'dlt_asc_{latest_date}.csv')
        df[latest_columns].to_csv(latest_file, index=False, encoding='utf-8')
        print(f"大乐透最新排好序数据已保存到 {latest_file}")


def get_dlt_order_data():
    """
    下载超级大乐透出球顺序数据（原始开奖顺序版）
    来源文件：dlt_asc.txt（带出球顺序）
    """
    url = 'http://data.17500.cn/dlt_asc.txt'
    columns = [
        '开奖期号', 'Date',
        'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7',
        "投注总额", "滚入下期一等奖池", "一等注数", "单注奖金_一等",
        "二等注数", "单注奖金_二等", "三等注数", "单注奖金_三等",
        "四等注数", "单注奖金_四等", "五等注数", "单注奖金_五等",
        "六等注数", "单注奖金_六等", "七等注数", "单注奖金_七等",
        "八等注数", "单注奖金_八等", "追加一等注数", "单注奖金_追加一等",
        "追加二等注数", "单注奖金_追加二等", "追加三等注数", "单注奖金_追加三等",
        '附加玩法投注总额', '一等奖注数', "单注奖金_附加", "r1", "r2"
    ]
    save_path = os.path.join('dlt', 'dlt_order_data.csv')
    df = download_and_save(url, columns, save_path)
    if df is not None:
        latest_date = df['Date'].iloc[-1]
        latest_columns = ['Date', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7']
        latest_file = os.path.join('dlt', f'dlt_order_{latest_date}.csv')
        df[latest_columns].to_csv(latest_file, index=False, encoding='utf-8')
        print(f"大乐透最新出球顺序数据已保存到 {latest_file}")


# ===================== 快乐8 =====================
# 定义快乐8完整表头，共128列
kl8_columns = (
        ['开奖期号', '开奖日期'] +
        ['排好序_' + str(i) for i in range(1, 21)] +
        ['出球顺序_' + str(i) for i in range(1, 21)] +
        [
            "本期销售金额", "选十玩法奖池金额",
            "选十中十注数", "单注奖金_十中十",
            "选十中九注数", "单注奖金_十中九",
            "选十中八注数", "单注奖金_十中八",
            "选十中七注数", "单注奖金_十中七",
            "选十中六注数", "单注奖金_十中六",
            "选十中五注数", "单注奖金_十中五",
            "选十中零注数", "单注奖金_十中零",
            "选九中九注数", "单注奖金_九中九",
            "选九中八注数", "单注奖金_九中八",
            "选九中七注数", "单注奖金_九中七",
            "选九中六注数", "单注奖金_九中六",
            "选九中五注数", "单注奖金_九中五",
            "选九中四注数", "单注奖金_九中四",
            "选九中零注数", "单注奖金_九中零",
            "选八中八注数", "单注金额_八中八",
            "选八中七注数", "单注金额_八中七",
            "选八中六注数", "单注金额_八中六",
            "选八中五注数", "单注金额_八中五",
            "选八中四注数", "单注金额_八中四",
            "选八中零注数", "单注金额_八中零",
            "选七中七注数", "单注金额_七中七",
            "选七中六注数", "单注金额_七中六",
            "选七中五注数", "单注金额_七中五",
            "选七中四注数", "单注金额_七中四",
            "选七中零注数", "单注金额_七中零",
            "选六中六注数", "单注金额_六中六",
            "选六中五注数", "单注金额_六中五",
            "选六中四注数", "单注金额_六中四",
            "选六中三注数", "单注金额_六中三",
            "选五中五注数", "单注金额_五中五",
            "选五中四注数", "单注金额_五中四",
            "选五中三注数", "单注金额_五中三",
            "选四中四注数", "单注金额_四中四",
            "选四中三注数", "单注金额_四中三",
            "选四中二注数", "单注金额_四中二",
            "选三中三注数", "单注金额_三中三",
            "选三中二注数", "单注金额_三中二",
            "选二中二注数", "单注金额_二中二",
            "选一中一注数", "单注金额_一中一"
        ]
)


def get_kl8_asc_data():
    """
    下载快乐8排好序数据（升序排列版）
    来源文件：kl8_asc.txt
    """
    url = 'http://data.17500.cn/kl8_asc.txt'
    save_path = os.path.join('kl8', 'kl8_asc_data.csv')
    df = download_and_save(url, kl8_columns, save_path)
    if df is not None:
        # 转换金额字段
        df["本期销售金额"] = df["本期销售金额"].apply(convert_currency)
        df["选十玩法奖池金额"] = df["选十玩法奖池金额"].apply(convert_currency)
        # 提取“排好序”号码部分（列3-22）
        latest_cols = ['排好序_' + str(i) for i in range(1, 21)]
        latest_file = os.path.join('kl8', 'kl8_asc_latest.csv')
        df[latest_cols].to_csv(latest_file, index=False, encoding='utf-8')
        print(f"快乐8最新排好序数据已保存到 {latest_file}")


def get_kl8_order_data():
    """
    下载快乐8出球顺序数据（原始开奖顺序版）
    来源文件：kl82_asc.txt（带出球顺序）
    """
    url = 'http://data.17500.cn/kl82_asc.txt'
    save_path = os.path.join('kl8', 'kl8_order_data.csv')
    df = download_and_save(url, kl8_columns, save_path)
    if df is not None:
        df["本期销售金额"] = df["本期销售金额"].apply(convert_currency)
        df["选十玩法奖池金额"] = df["选十玩法奖池金额"].apply(convert_currency)
        # 提取“出球顺序”号码部分（列23-42）
        latest_cols = ['出球顺序_' + str(i) for i in range(1, 21)]
        latest_file = os.path.join('kl8', 'kl8_order_latest.csv')
        df[latest_cols].to_csv(latest_file, index=False, encoding='utf-8')
        print(f"快乐8最新出球顺序数据已保存到 {latest_file}")


# ===================== 双色球 =====================
def get_ssq_asc_data():
    """
    下载双色球排好序数据（升序排列版）
    来源文件：ssq_asc.txt
    """
    url = 'http://data.17500.cn/ssq_asc.txt'
    columns = [
        '开奖期号', 'Date',
        's1', 's2', 's3', 's4', 's5', 's6', 's7',
        'xs1', 'xs2', 'xs3', 'xs4', 'xs5', 'xs6', 'xs7',
        "投注总额", "奖池金额", "一等注数", "一等金额",
        "二等注数", "二等金额", "三等注数", "三等金额",
        "四等注数", "四等金额", "五等注数", "五等金额", "六等注数"
    ]
    save_path = os.path.join('ssq', 'ssq_asc_data.csv')
    df = download_and_save(url, columns, save_path)
    if df is not None:
        latest_date = df['Date'].iloc[-1]
        latest_cols = ['Date', 's1', 's2', 's3', 's4', 's5', 's6', 's7']
        latest_file = os.path.join('ssq', f'ssq_asc_{latest_date}.csv')
        df[latest_cols].to_csv(latest_file, index=False, encoding='utf-8')
        print(f"双色球最新排好序数据已保存到 {latest_file}")


def get_ssq_order_data():
    """
    下载双色球出球顺序数据（原始开奖顺序版）
    来源文件：ssq_asc.txt（正序文件已保留出球顺序）
    """
    url = 'http://data.17500.cn/ssq_asc.txt'
    columns = [
        '开奖期号', 'Date',
        's1', 's2', 's3', 's4', 's5', 's6', 's7',
        "投注总额", "奖池金额", "一等注数", "一等金额",
        "二等注数", "二等金额", "三等注数", "三等金额",
        "四等注数", "四等金额", "五等注数", "五等金额", "六等注数"
    ]
    save_path = os.path.join('ssq', 'ssq_order_data.csv')
    df = download_and_save(url, columns, save_path)
    if df is not None:
        latest_date = df['Date'].iloc[-1]
        latest_cols = ['Date', 's1', 's2', 's3', 's4', 's5', 's6', 's7']
        latest_file = os.path.join('ssq', f'ssq_order_{latest_date}.csv')
        df[latest_cols].to_csv(latest_file, index=False, encoding='utf-8')
        print(f"双色球最新出球顺序数据已保存到 {latest_file}")


def convert_currency(value):
    """
    将类似 '1,234,567.89' 格式的金额字符串转换为整数
    """
    try:
        return int(float(value.replace(",", "")))
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="彩票数据下载器：下载超级大乐透、快乐8、双色球的历史数据")
    parser.add_argument('--data_type', choices=['asc', 'order'], default='order',
                        help="数据类型：asc 表示排好序数据，order 表示出球顺序数据（默认）")
    args = parser.parse_args()

    ensure_directories()

    if args.data_type == 'asc':
        print("下载排好序数据（升序排列）...")
        get_dlt_asc_data()
        get_kl8_asc_data()
        get_ssq_asc_data()
    else:
        print("下载出球顺序数据（原始开奖顺序）...")
        get_dlt_order_data()
        get_kl8_order_data()
        get_ssq_order_data()


if __name__ == "__main__":
    main()

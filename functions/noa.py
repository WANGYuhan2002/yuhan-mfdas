# === dasmf path bootstrap (auto-added; do not remove) ===
import os as _dasmf_os, sys as _dasmf_sys
_dasmf_p = _dasmf_os.path.abspath(__file__)
while _dasmf_p != _dasmf_os.path.dirname(_dasmf_p) and not _dasmf_os.path.exists(_dasmf_os.path.join(_dasmf_p, '.dasmf_root')):
    _dasmf_p = _dasmf_os.path.dirname(_dasmf_p)
PROJECT_ROOT = _dasmf_p
if _dasmf_os.path.join(PROJECT_ROOT, 'functions') not in _dasmf_sys.path:
    _dasmf_sys.path.insert(0, _dasmf_os.path.join(PROJECT_ROOT, 'functions'))
if PROJECT_ROOT not in _dasmf_sys.path:
    _dasmf_sys.path.insert(0, PROJECT_ROOT)
# === end bootstrap ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.lines import Line2D
import os
import geopandas
import contextily as cx
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from geopy.distance import geodesic
from datetime import datetime
import re
import requests
from io import StringIO
import io


def extract_station_codes(ims_filepath):
    """
    从IMS格式的地震公报文件中提取所有唯一的台站代码。
    这个函数通过正则表达式匹配那些以台站代码开头的行（即震相到时记录）。

    参数:
    ims_filepath (str): IMS文件的路径。

    返回:
    set: 包含所有唯一台站代码的集合。
    """
    station_codes = set()
    station_line_pattern = re.compile(r'^([A-Z0-9]{3,5})\s+\d+\.\d+.*')
    
    try:
        with open(ims_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                match = station_line_pattern.match(line)
                if match:
                    station_codes.add(match.group(1))
    except FileNotFoundError:
        print(f"错误：找不到文件 {ims_filepath}")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        
    return station_codes

def load_station_inventory(network_codes=['HL', 'HP', 'HT', '5B']):
    """
    从EIDA FDSN web service加载官方台站inventory。
    这是获取精确台站位置的权威方法。
    
    [已修正] 此函数已更新，以正确解析FDSN的文本输出格式。
    旧版本错误地将表头行作为注释忽略，导致列名解析失败。

    参数:
    network_codes (list): 要查询的台网代码列表。
                          默认为希腊地区的主要台网。

    返回:
    pandas.DataFrame: 包含台站信息的DataFrame，列为['code', 'lat', 'lon']。
    """
    networks_str = ",".join(network_codes)
    print(f"正在从官方FDSN源加载 {networks_str} 台网的台站inventory...")
    url = f"http://eida.gein.noa.gr/fdsnws/station/1/query?network={networks_str}&level=station&format=text"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # --- [修正逻辑] ---
        # FDSN的文本格式将表头放在一个以'#'开头的行中。
        # 我们需要手动解析这个表头，然后读取没有注释的数据。
        
        all_lines = response.text.strip().splitlines()
        
        # 提取表头
        header_line = None
        for line in all_lines:
            if line.startswith('#'):
                header_line = line
                break
        
        if not header_line:
            raise ValueError("在FDSN响应中未找到表头行。")
            
        column_names = [col.strip() for col in header_line.lstrip('#').split('|')]
        
        # 提取数据行 (所有不以'#'开头的行)
        data_lines = [line for line in all_lines if not line.startswith('#')]
        
        if not data_lines:
            print("警告: FDSN查询未返回任何台站数据。")
            return pd.DataFrame(columns=['code', 'lat', 'lon'])
            
        # 将数据行组合成一个字符串，供pandas读取
        csv_data = '\n'.join(data_lines)
        data_io = io.StringIO(csv_data)
        
        # 读取CSV，不使用文件的第一行作为表头，而是手动指定我们解析的表头
        df = pd.read_csv(data_io, sep='|', header=None, names=column_names)
        
        # 清理列名和数据
        df.columns = df.columns.str.strip()
        df_coords = df[['Station', 'Latitude', 'Longitude']].copy()
        df_coords.rename(columns={
            'Station': 'code',
            'Latitude': 'lat',
            'Longitude': 'lon'
        }, inplace=True)
        df_coords['code'] = df_coords['code'].str.strip()
        
        print(f"成功加载 {len(df_coords)} 个台站的精确位置。")
        return df_coords
        
    except requests.exceptions.RequestException as e:
        print(f"错误：无法从FDSN URL加载台站数据: {e}")
    except Exception as e:
        print(f"处理台站数据时发生错误: {e}")
        
    print("将返回一个空的DataFrame。")
    return pd.DataFrame(columns=['code', 'lat', 'lon'])
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
import re
import os
from datetime import datetime

def extract_station_info(file_path, station_name):
    """
    Fit for file format in IMS1.0

    Given:  the earthquake info list at certain format
            certain station name
            
    Return: the earthquake record list of the station, containing: ...

    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read()

    # 按 STOP 分割不同的地震事件
    events = data.split("\n\nSTOP\n\n")

    Record_dict_ls = []
    count = 0
    for event in events:
        if station_name in event: # 如果事件中包含指定的台站名
            count += 1
            # print(f"\n----Find {count}th earthquake recorded by station {station_name}!")
            
            #*1-----search 1---- event
            event_info_match = re.search(r'(\d{4}/\d{2}/\d{2})\s+(\d{2}:\d{2}:\d{2}\.\d+).*?(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)', event)
            date, time, rms, lat, lon, depth = event_info_match.groups()
            EQ_time = datetime.strptime(f"{date} {time}", "%Y/%m/%d %H:%M:%S.%f")
            # print(f" Date: {date}, Time: {time}, Lat: {lat}, Lon: {lon}, Depth: {depth} km")
            
            #*2------search 2---- event
            mag_info_match = re.search(r'ML\s+(\d+\.\d+)', event)
            magnitude = mag_info_match.group(1)
            # print(f"Magnitude: {magnitude}")
            

            #*3-----search 3----- this can generate multiple lines
            sta_name_pattern = fr'{station_name}\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\S+)\s+(\d+:\d+:\d+\.\d+)'
            sta_matches = re.findall(sta_name_pattern, event)

            for ii,sta_match in enumerate(sta_matches):
                Dist, EvAz, Phase, Timestr = sta_match
                Time = datetime.strptime(f"{date} {Timestr}", "%Y/%m/%d %H:%M:%S.%f")
                # print(f"<Record{ii}> Station Name: {station_name}, Distance: {Dist}, Azimuth: {EvAz}, Phase: {Phase}, Arrival Time: {Time}")
            
            
            #*4-----create dict() for each 'record'-------
            for ii,sta_match in enumerate(sta_matches):
                Dist, EvAz, Phase, Timestr = sta_match
                tmpRecord_dict = {
                    'EQ_time': EQ_time,
                    'EQ_latitude': lat,
                    'EQ_longitude': lon,
                    'EQ_depth': depth,
                    'EQ_magnitude': magnitude,
                    'STA_name': station_name,
                    'STA_distance': Dist,
                    'STA_azimuth': EvAz,
                    'STA_phase': Phase,
                    'STA_arrival_time': Time
                }
                Record_dict_ls.append(tmpRecord_dict)
    
    print(f"\n>>>Total {count} earthquakes recorded by station {station_name}!")
    
    return Record_dict_ls

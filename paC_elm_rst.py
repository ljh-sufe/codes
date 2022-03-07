# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 10:57:30 2019

@author: 41409
"""

from urllib.parse import urlencode
import pandas as pd
import requests
import random
import time
import re


def paC_elm_rst(id_):
    url = 'https://h5.ele.me/pizza/shopping/restaurants/' + id_ +'/batch_shop?'
    cookie = 'ubt_ssid=l577qi6vfgxkac6a0a83xx10x53avr52_2019-08-13; _utrace=e618683e3d7650b828c14300e373d062_2019-08-13; cna=gBTZFQ7I1CMCAWVUl+UFJ5lo; perf_ssid=kts4f36m2wcl2th7eqypi8y6urf5h8l5_2019-08-13; ut_ubt_ssid=pms66uivqy7bn2wejbvjhd5wd6qimb0x_2019-08-13; _bl_uid=b3j0qz309178I7hIb5Rv17LigRRk; track_id=1565664987|03bcd1fa3f92d97414c4b004e43dcf7c83e1a52d21ed5f1b79|456037998db7c0bea59f3554046213b5; USERID=28445125; tzyy=e9f0c1437e35eab863e6adb9186ccd2f; UTUSER=28445125; SID=le5gT4YGOdlXd7qaHpRjjF9uawlUjuQRH1tA; ZDS=1.0|1565831326|rstuKRvC2wVKn2Vhcnb6rQR/ywb5yg881UQFa4lQYQQrpDKS875VfiOvRo/styZm; __wpkreporterwid_=dc2db657-eb9d-480d-04a9-ade6bb224e95; pizza73686f7070696e67=soFDFDje30wI1kI_0157nEy8srybQMcK9_tTP7dfs9tNniOUmY95kaSybJxPc_r1; l=dBaVI4qVqi6Q8eKUBOCanurza77OSIRvmuPzaNbMi_5Q56Ys4kBOkIpLKFv62jWf9dLB45113t29-etkieWJ06Xj0UteYxDc.; isg=BK2te5ndjMzHPGiyefo-3b_kvEmLHuvaX_7EMO-y6cSzZs0Yt1rxrPswUXolUvmU'
    parm_head = {
            'user_id': '28445125',
            'code': '0.7947755744403053'
            }
    parm_mid = 'extras=%5B%22activities%22%2C%22albums%22%2C%22license%22%2C%22identification%22%2C%22qualification%22%5D'
    parm_tail = {
            'terminal': 'h5',
            'latitude': '31.30553',
            'longitude': '121.49465'
            }
    ajax_url = url + urlencode(parm_head) + parm_mid + urlencode(parm_tail)
    headers = {
            'cookie': cookie,
            'referer': 'https://h5.ele.me/shop/',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.100 Safari/537.36'
            }
    response = requests.get(ajax_url, headers=headers)
    json = response.json()
    return json

def ana_1ddjson(json):
    '''
    analyze json
    return data_rst, data_spec
    '''
    data_rst = pd.DataFrame()
    data_spec = pd.DataFrame()
    templist_rst = []
    templist_spec = []
    
    inf_food = ['lowest_price', 'materials', 'rating', 'rating_count', 'satisfy_count', 'satisfy_rate', 'month_sales']
    inf_spec = ['price', 'rating', 'count', 'recent_rating', 'satisfy', 'recent_popularity']        
    menu = json['menu']
    for i in range(1, len(menu)):
        type_ = menu[i]
        foods = type_['foods']
        num_of_foods = len(foods)
        for j in range(0, num_of_foods):
            item = foods[j]        
            templist_rst.append(item['name'])
            templist_rst.append(json['rst']['name'][4:-1])
            templist_rst.append(type_['name'])
            for inf_ in inf_food:
                templist_rst.append(item[inf_])
            data_rst = data_rst.append([templist_rst], ignore_index = True)
            templist_rst = []         

            spec = item['specfoods']
            for m in range(0, len(spec)):
                specfood = spec[m]
                templist_spec.append(specfood['name'])
                templist_spec.append(json['rst']['name'][4:-1])
                for inf__ in inf_spec:
                    templist_spec.append(specfood[inf__])
                
                searcharea = specfood['specs'][0]['value']     
                if re.search('冰淇淋', searcharea):
                    templist_spec.append('冰淇淋')
                elif re.search('布丁', searcharea):
                    templist_spec.append('布丁')
                elif re.search('奶霜', searcharea):
                    templist_spec.append('奶霜')
                elif re.search('燕麦', searcharea):
                    templist_spec.append('燕麦')
                elif re.search('免费配料', searcharea):
                    templist_spec.append('免费配料')
                elif re.search('咖啡冻', searcharea):
                    templist_spec.append('咖啡冻')
                else:
                    templist_spec.append(searcharea)        
                data_spec = data_spec.append([templist_spec], ignore_index = True)
                templist_spec = []
    return data_rst, data_spec

def get_idd(res_ID):
    '''
    根据id爬取店家页面
    return 整合到一起的数据data_temp_rst, data_temp_spec带columns标签
    '''
    i = 0
    for id in res_ID:
        json = paC_elm_rst(id)    #爬取店家页面，得到json
        data_rst, data_spec = ana_1ddjson(json)    #得到data_rst, data_spec
        #把每家店的data_rst, data_spec拼接
        if i == 0:
            data_temp_rst = data_rst
            data_temp_spec = data_spec
        else:
            data_temp_rst = pd.concat([data_temp_rst, data_rst], ignore_index = True)
            data_temp_spec = pd.concat([data_temp_spec, data_spec], ignore_index = True)
        print('ID:' + res_ID[i] + '爬取成功，店铺名为：' + data_rst.loc[0, 1])
        i = i + 1
        time.sleep(random.randint(2,5))
    #添加columns标签
    columns_rst = ['name', 'restaurant_name', 'type_name', 'price', 'main_material', 'rating', 'rating_count', 'satisfy_count', 'satisfy_rate', 'month_sales']
    data_temp_rst.columns = columns_rst
    columns_spec = ['name', 'restaurant_name', 'price', 'rating', 'count', 'recent_rating', 'satisfy', 'month_sales', 'material_added']
    data_temp_spec.columns = columns_spec
    return data_temp_rst, data_temp_spec

def ana_cocojson(json):
    data_rst = pd.DataFrame()
    data_spec = pd.DataFrame()
    templist_rst = []
    templist_spec = []
    
    inf_food = ['lowest_price', 'description', 'rating', 'rating_count', 'satisfy_count', 'satisfy_rate', 'month_sales']
    inf_spec = ['price', 'rating', 'count', 'recent_rating', 'satisfy', 'recent_popularity']
    menu = json['menu']
    for i in range(1, len(menu)):
        type_ = menu[i]
        foods = type_['foods']
        num_of_foods = len(foods)
        for j in range(0, num_of_foods):
            item = foods[j]
            templist_rst.append(item['name'])
            templist_rst.append(json['rst']['name'][7:-1])
            templist_rst.append(type_['name'])
            for inf_ in inf_food:
                templist_rst.append(item[inf_])
            data_rst = data_rst.append([templist_rst], ignore_index = True)
            templist_rst = []
            
            spec = item['specfoods']
            for m in range(0, len(spec)):
                specfood = spec[m]
                templist_spec.append(specfood['name'])
                templist_spec.append(json['rst']['name'][7:-1])
                for inf__ in inf_spec:
                    templist_spec.append(specfood[inf__])
                if len(specfood['specs']) == 0:
                    templist_spec.append('常规')
                else:
                    templist_spec.append(specfood['specs'][0]['value'])      
                data_spec = data_spec.append([templist_spec], ignore_index = True)
                templist_spec = []
    return data_rst, data_spec

def get_coco(res_ID):
    '''
    根据id爬取店家页面
    return 整合到一起的数据data_temp_rst, data_temp_spec带columns标签
    '''
    i = 0
    for id in res_ID:
        json = paC_elm_rst(id)    #爬取店家页面，得到json
        data_rst, data_spec = ana_cocojson(json)    #得到data_rst, data_spec
        #把每家店的data_rst, data_spec拼接
        if i == 0:
            data_temp_rst = data_rst
            data_temp_spec = data_spec
        else:
            data_temp_rst = pd.concat([data_temp_rst, data_rst], ignore_index = True)
            data_temp_spec = pd.concat([data_temp_spec, data_spec], ignore_index = True)
        print('ID:' + res_ID[i] + '爬取成功，店铺名为：' + data_rst.loc[0, 1])
        i = i + 1
        time.sleep(random.randint(2,5))
    #添加columns标签
    columns_rst = ['name', 'restaurant_name', 'type_name', 'price', 'main_material', 'rating', 'rating_count', 'satisfy_count', 'satisfy_rate', 'month_sales']
    data_temp_rst.columns = columns_rst
    columns_spec = ['name', 'restaurant_name', 'price', 'rating', 'count', 'recent_rating', 'satisfy', 'month_sales', 'material_added']
    data_temp_spec.columns = columns_spec
    return data_temp_rst, data_temp_spec
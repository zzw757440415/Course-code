# !/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@func: 爬取微博上独山县的相关评论
@author: Ziwang Zhao
@file: spyder_weibo.py
@time: 2020/7/16
"""
import requests
import re
import time


def get_one_page(url):
    headers = {
        'User-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36',
        'Host': 'weibo.cn',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Cookie': '_T_WM=26525842305; WEIBOCN_WM=3333_2001; '
                  'ALF=1597418498; SUB=_2A25yC29VDeRhGeBK6lYZ-CbKzjiI'
                  'HXVR9HEdrDV6PUNbktCOLXHBkW1NR_mMCVtei7oyyVNGhcUVEn4bEejfLynK;'
                  ' SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9WhLJ3dVIDXzkwgQQh95qbFa5JpX5KzhUgL'
                  '.FoqXeKBR1hncSKB2dJLoI7DAIPUydJMXS0.c; SUHB=0cS3oxSXutUyzi; SSOLoginState=1594826501;'
                  ' MLOGIN=1; M_WEIBOCN_PARAMS=oid%3D4525952443578662%26luicode%3D20000061%26lfid%3D4525952443578662',
        'DNT': '1',
        'Connection': 'keep-alive'
    }
    response = requests.get(url, headers=headers, verify=False)  # 利用requests.get命令获取网页html
    if response.status_code == 200:  # 状态为200即为爬取成功
        return response.text  # 返回值为html文档，传入到解析函数当中
    return None


def parse_one_page(html):  # 解析html并存入到文档result.txt中
    pattern = re.compile('<span class="ctt">.*?</span>', re.S)
    items = re.findall(pattern, html)
    result = str(items)
    with open('Comment_weibo.txt', 'a', encoding='utf-8') as fp:
        for i in re.findall((r", \'<span class=\"ctt\">(.*?)</span>\',"), result):
            if 'img' not in i and 'href' not in i:
                print(i)
                fp.write(i + '\n')
        # fp.write('\n')


for i in range(1, 24):
    url = "https://weibo.cn/comment/JbczW4wvt?uid=1988800805&rl=1&page="+str(i)
    html = get_one_page(url)
    print('正在爬取第 %d 页评论' % (i))
    parse_one_page(html)
    time.sleep(0.5)


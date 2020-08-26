# !/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@func: 爬取哔哩哔哩《睡前消息140---独山县》的视频评论
@author: Ziwang Zhao
@file: spyder_bilibili.py
@time: 2020/7/16
"""
import requests
import json


def dec(x):
    '''BV号转AV号'''
    r = 0
    for i in range(6):
        r += tr[x[s[i]]] * 58 ** i
    return (r - add) ^ xor


def enc(x):
    '''BV号转AV号'''
    x = (x ^ xor) + add
    r = list('BV1  4 1 7  ')
    for i in range(6):
        r[s[i]] = table[x // 58 ** i % 58]
    return ''.join(r)


def get_text(bv, pn, sort, i):
    '''
    获取Bilibili视频评论
    :param bv: 视频BV号
    :param pn: 获取评论页数
    :param sort: 排序种类 0是按时间排序 2是按热度排序
    :return:
    '''
    headers = {
        'User-Agent': 'XXX'
    }
    oid = dec(bv)
    fp = open('Comment_bilibili.txt', "w", encoding="UTF-8")
    while True:
        url = f'https://api.bilibili.com/x/v2/reply?pn={pn}&type=1&oid={oid}&sort={sort}'
        reponse = requests.get(url, headers=headers)
        a = json.loads(reponse.text)
        if pn == 1:
            count = a['data']['page']['count']
            size = a['data']['page']['size']
            page = count // size + 1
        for b in a['data']['replies']:
            panduan = 0
            str1 = ''
            str_list = list(b['content']['message'])
            for x in range(len(str_list)):
                if str_list[x] == '[':
                    panduan = 1
                if panduan != 1:
                    str1 = str1 + str_list[x]
                if str_list[x] == ']':
                    panduan = 0
            fp.write(str1)
            print(str1, '\n', '-' * 10)
            i = i + 1
        if pn != page:
            pn += 1
        else:
            fp.close()
            break


if __name__ == '__main__':
    bv = 'BV1rt4y1Q74C'
    i = 1
    tr = {}
    table = 'fZodR9XQDSUm21yCkr6zBqiveYah8bt4xsWpHnJE7jL5VG3guMTKNPAwcF'
    for i in range(58):
        tr[table[i]] = i
    s = [11, 10, 3, 8, 4, 6]
    xor = 177451812
    add = 8728348608
    get_text(bv, 1, 2, i)

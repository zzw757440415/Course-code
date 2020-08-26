#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@func:
@author: Ziwang Zhao
@file: processing.py
@time: 2020/7/16
"""
import jieba
import re
import time
import jieba.analyse as analyse
import seaborn as sns
from snownlp import SnowNLP
import unicodedata
import jiagu
import heapq
import collections
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS


def get_comment():
    '''获取评论'''
    com_bili, com_weibo = [], []
    pattern1 = re.compile(u'\t|\n|\.|-|:|;|\)|\(|\?|"')  # 去除字符串内的多余标点符号
    pattern2 = re.compile("[A-Za-z0-9\!\%\[\]\,\。]")  # 去除字符串内的多余数字和字母
    with open('NetworkSpyder/Comment_bilibili.txt', 'r', encoding='utf-8') as bi:
        rows_bi = bi.readlines()
        for row_bi in rows_bi:
            data_bi = re.sub(pattern1, '', row_bi)
            data_bi = re.sub(pattern2, '', data_bi)
            data_bi = unicodedata.normalize('NFKC', data_bi)
            com_bili.append([data_bi])
        # print(com_bili[:5])
    with open('NetworkSpyder/Comment_weibo.txt', 'r', encoding='utf-8') as wei:
        rows_wei = wei.readlines()
        for row_wei in rows_wei:
            data_wei = re.sub(pattern1, '', row_wei)
            data_wei = re.sub(pattern2, '', data_wei)
            data_wei = unicodedata.normalize('NFKC', data_wei)
            com_weibo.append([data_wei])
        # print(com_weibo[:5])

    return com_bili, com_weibo


def get_chinese(string):
    '''
    去除无用符号, 仅保留字符串内的中文, 返回分词列表
    :param string: 文本字符串
    :return: 文本分词后的列表
    '''
    pattern = re.compile(u'\t|\n|\.|-|:|;|\)|\(|\?|\？|"')
    data = re.sub(pattern, '', string)
    data = re.sub("[A-Za-z0-9\!\%\[\]\,\。]", "", data)
    seg_list_exact = jieba.cut(data, cut_all=False)
    object_list = []
    #  自定义常见去除词库
    remove_words = [u'的', u'，', u'和', u'是', u'随着', u'对于', u'对', u'等', u'能', u'都', u'。', u' ', u'、', u'中', u'在', u'了',
                    u'通常', u'如果', u'我们', u'需要', u'？', u'我', u'！', u'这', u'也', u'就', u'不', u'啊', u'一个', u'有', u'还',
                    u'以', u'吗', u'人', u'吧', u'这些', u'真的', u'这么', u'挺', u'他', u'没', u'这样', u'可以', u'去', u'好', u'看',
                    u'这个', u'你', u'很', u'什么', u'多', u'没有', u'上', u'呢', u'地方', u'就是', u'知道', u'给', u'搞', u'想', u'个',
                    u'说', u'要', u'还是', u'怎么', u'不是', u'多少', u'把', u'到', u'被', u'来', u'感觉', u'大', u'会', u'做', u'太', u'现在'
                    u'“', "”", u'不错', u'呀', u'应该', u'...', u'但是', u'看到', u'觉得']
    for word in seg_list_exact:
        if word not in remove_words:
            object_list.append(word)
    word_counts = collections.Counter(object_list)

    return word_counts


def creat_wordcloud(com_bili, com_weibo):
    '''对评论制作词云'''
    com_bili_str, com_wei_str = '', ''
    for i in com_bili:
        com_bili_str += i[0]
    for j in com_weibo:
        com_wei_str += j[0]
    com_bili_counts = get_chinese(com_bili_str)
    com_wei_counts = get_chinese(com_wei_str)
    # 绘制词云
    background_bilibili = plt.imread('./images/bilibili.jpg')
    background_weibo = plt.imread('./images/weibo.jpg')
    wc_bilibili = WordCloud(background_color='white', mask=background_bilibili, scale=2,
                            max_words=200, stopwords=STOPWORDS, font_path='C:/Windows/Fonts/STLITI.TTF',
                            max_font_size=150, random_state=50, width=1000, height=600)
    wc_weibo = WordCloud(background_color='white', mask=background_weibo, scale=2,
                         max_words=200, stopwords=STOPWORDS, font_path='C:/Windows/Fonts/STLITI.TTF',
                         max_font_size=100, random_state=50, width=1000, height=600)
    wc_bilibili.generate_from_frequencies(com_bili_counts)
    wc_weibo.generate_from_frequencies(com_wei_counts)
    # 保存词云
    wc_bilibili.to_file('images/wordcloud_bilibili.png')
    wc_weibo.to_file('images/wordcloud_weibo.png')
    # 可视化词云
    plt.axis('off')
    plt.imshow(wc_bilibili)
    plt.show()
    plt.axis('off')
    plt.imshow(wc_weibo)
    plt.show()


def count(com_bili, com_weibo):
    '''分析评论长度'''
    count_bili, count_wei = np.array([len(i[0]) for i in com_bili]), np.array([len(i[0]) for i in com_weibo])
    mean_bili, median_bili = np.mean(count_bili), np.median(count_bili)  # 获取哔哩哔哩评论长度的均值、中位数
    mean_wei, median_wei = np.mean(count_wei), np.mean(count_wei)  # 获取微博评论长度的均值、中位数
    print('哔哩哔哩评论长度均值:{}, 中位数:{}.'.format(int(mean_bili), int(median_bili)))
    print('微博评论长度均值:{}, 中位数:{}.'.format(int(mean_wei), int(median_wei)))


def snow_sentiment(com_bili, com_weibo):
    '''使用SnowNLP进行情感分类, 绘制情感分布图'''
    def get_counter(comment):
        '''获取每条评论的情感'''
        sentiment = []
        for i in comment:
            if len(i[0]):
                res = SnowNLP(i[0]).sentiments  # 获取正面情绪的概率
                # 将情绪分为 5 个级别
                if res > 0.8:
                    sentiment.append(2)
                elif res > 0.6:
                    sentiment.append(1)
                elif res > 0.4:
                    sentiment.append(0)
                elif res > 0.2:
                    sentiment.append(-1)
                else:
                    sentiment.append(-2)
        return sentiment

    sentiment_bili, sentiment_wei = get_counter(com_bili), get_counter(com_weibo)
    counter_bili, counter_wei = collections.Counter(sentiment_bili),  collections.Counter(sentiment_wei)
    # print(counter_bili)  # Counter({2: 437, -2: 176, 0: 109, -1: 98, 1: 79})
    # print(counter_wei)  # Counter({2: 254, -2: 236, 1: 188, 0: 187, -1: 146})
    data_bili = [counter_bili[2], counter_bili[1], counter_bili[0], counter_bili[-1], counter_bili[-2]]
    data_wei = [counter_wei[2], counter_wei[1], counter_wei[0], counter_wei[-1], counter_wei[-2]]
    laebls = ['积极', '比较积极', '中立', '比较消极', '消极']
    indic = [0.1, 0, 0, 0, 0]
    plt.title("哔哩哔哩评论 SnowNLP情感分析结果")
    plt.pie([i/sum(data_bili) for i in data_bili], labels=laebls, shadow=True, explode=tuple(indic))
    plt.show()
    plt.title("微博评论 SnowNLP情感分析结果")
    plt.pie([j/sum(data_wei) for j in data_wei], labels=laebls, shadow=True, explode=tuple(indic))
    plt.show()


def get_view(comments, label):
    '''统计各种情感倾向的观点并可视化'''
    comment_str = ''
    for i in comments:
        if len(i[0]) and '\u4e00' <= i[0][-1] <= '\u9fa5':
            comment_str += i[0]+'。'
        else:
            comment_str += i[0]
    summarize = jiagu.summarize(comment_str, 3)  # 以文本摘要的形式生成三种观点
    key_word = analyse.extract_tags(comment_str, topK=5, allowPOS=('ns', 'n', 'vn', 'nz', 'nt', 'nr'))  # TF-IDF生成关键词
    print(label, '关键语句', summarize)
    print(label, '关键词', key_word)

    # 可视化观点和关键词
    string = min(summarize, key=len, default='')
    w = WordCloud(background_color='white', scale=2, stopwords=STOPWORDS,
                  font_path='C:/Windows/Fonts/STLITI.TTF', random_state=20)
    w.generate(string)
    w.to_file('images/{}.png'.format(label))
    plt.axis('off')
    plt.title(label)
    plt.imshow(w)
    plt.show()


def LSTM_sentiment(com_bili, com_weibo):
    '''利用预训练 Bi-LSTM 进行情感分类, 并分析主要观点 '''
    def get_probability(comments, label, n):
        sentiment = []
        pos_num, neg_num = [], []
        for i in comments:
            if len(i[0]):
                res = jiagu.sentiment(i[0])  # tuple(str, float)
                sentiment.append(res)
                if res[0] == 'positive':
                    pos_num.append(res[1])
                else:
                    neg_num.append(res[1])
        # 绘制直方图和核密度估计曲线
        sns.distplot(pos_num, bins=80, hist=True, norm_hist=True, rug=False)
        sns.distplot(neg_num, bins=80, hist=True, norm_hist=True, rug=False)
        plt.legend([
            "{} 积极: {:.1f}%".format(label, 100 * len(pos_num) / (len(pos_num) + len(neg_num))),
            "{} 消极: {:.1f}%".format(label, 100 * len(neg_num) / (len(pos_num) + len(neg_num)))
        ])
        plt.xlabel('评价置信度')
        plt.ylabel('频数')
        plt.show()

        # 求高置信度观点
        pos_view_index = list(map(pos_num.index, heapq.nlargest(n, pos_num)))
        neg_view_index = list(map(neg_num.index, heapq.nlargest(n, neg_num)))
        pos_view = [comments[i] for i in pos_view_index]
        neg_view = [comments[i] for i in neg_view_index]

        return pos_view, neg_view

    pos_view_bili, neg_view_bili = get_probability(com_bili, "哔哩哔哩", 30)
    pos_view_wei, neg_view_wei = get_probability(com_weibo, "微博", 30)

    get_view(pos_view_wei, '微博---积极')
    get_view(pos_view_bili, '哔哩哔哩---积极')
    get_view(neg_view_bili, '哔哩哔哩---消极')
    get_view(neg_view_wei, '微博---消极')


def main():
    start = time.time()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    com_bili, com_weibo = get_comment()  # 获取评论

    print('开始生成词云')
    creat_wordcloud(com_bili, com_weibo)  # 生成词云
    print('开始分析评论长度')
    count(com_bili, com_weibo)  # 分析评论长度
    print('使用SnowNLP进行情感分类')
    snow_sentiment(com_bili, com_weibo)  # 使用SnowNLP进行情感分类, 绘制情感分布图
    print('使用Bi-LSTM进行情感分类')
    LSTM_sentiment(com_bili, com_weibo)  # 使用Bi-LSTM进行情感分类, 绘制情感分布图
    print('Finish, time: ', time.time() - start)


if __name__ == '__main__':
    main()

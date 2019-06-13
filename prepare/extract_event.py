# -*- coding: utf-8 -*-
import json
import pprint
import re
import threading

import numpy as np
import requests
from constants import EVENT_BASE, EVENT_TRAIN

EVENT_TRAIN = "../data/all/1000/refine_all_events_15to18.txt"

url = ['http://127.0.0.1:9612/ltp', 'http://127.0.0.1:9613/ltp', 'http://127.0.0.1:9614/ltp',
       'http://127.0.0.1:9615/ltp']
# url = ['http://127.0.0.1:9612/ltp']
year_scale = ["2015", "2016", "2017", "2018"]
count = 0
filter_part = ['wp', 'q', 'm', 'd']
filter_part_weak = ['wp', 'q', 'm']
all_events = set()
for line in open(EVENT_BASE, encoding="utf8"):
    try:
        title, date = line.strip().split('\t')
        date = date.strip().split()[0]
        year = date.split('-')[0]

        # only parse the year in year_scale
        if year not in year_scale:
            continue

        if title != "" and date != "":
            all_events.add((title, date))
        # if len(all_events) > 100:
        #     break
    except:
        pass

all_events = list(all_events)
print(len(all_events))
np.random.shuffle(all_events)

sample_events = dict()
number = 0
for event in all_events:
    date = event[1]
    title = event[0]
    if date in sample_events:
        if len(sample_events[date]) < 1000:
            sample_events[date].append(title)
            number += 1
    else:
        sample_events[date] = [title]
        number += 1
print("all sample number:", number)

events = []
keys = sample_events.keys()
for key in keys:
    date_event = sample_events[key]
    for item in date_event:
        events.append([item, key])

print("events:", len(events))
# events = events[0:100]
# print(events)

lock = threading.Lock()


def process(num):
    def thread_process(i, num, count):
        write_sub = []
        events_sub = events[i * int(len(events) / num):(i + 1) * int(len(events) / num)]
        for event in events_sub:
            try:
                # remove comma between numbers
                raw_title = event[0]
                remove_comma = re.compile(r'\d+,\d+?')
                for c in remove_comma.finditer(raw_title):
                    cc = c.group()
                    raw_title = raw_title.replace(cc, cc.replace(',', ''))

                raw_title = re.split(r'\。|？|！| |\?|!|:|：', raw_title)
                date = event[1]
                for title in raw_title:
                    if len(title) < 5:
                        continue
                    data = {'s': title, 'x': 'n', 't': 'srl'}
                    res = requests.post(url[i], data)
                    paras = json.loads(res.content.decode('utf-8'))[0]
                    # print(title)
                    # pprint.pprint(paras)
                    for sent in paras:
                        for item in sent:
                            if item['arg'] and item['pos'] == 'v':
                                p = [item['id']]
                                a0, a1, a0_com, a1_com, SBV, VOB, p_ADV = [], [], [], [], [], [], []  # None, None
                                all_id = []  # reduce repeat
                                # a0_com_list, a1_com_list = [], []
                                # p = item['cont']
                                for arg in item['arg']:
                                    # a = ' '.join([
                                    #     sent[i]['cont'] for i in range(arg['beg'], arg['end'] + 1)
                                    # ])
                                    for id in range(arg['beg'], arg['end'] + 1):
                                        if sent[id]['pos'] not in filter_part_weak:
                                            # tmp.append(sent[id]['cont'])
                                            if 'A0' == arg['type']:
                                                a0.append(id)
                                                all_id.append(id)
                                            elif 'A1' == arg['type']:
                                                a1.append(id)
                                                all_id.append(id)
                                            elif 'ADV' == arg['type']:
                                                p_ADV.append(id)
                                                all_id.append(id)
                                    # a = ' '.join(tmp)
                                    # if 'A0' == arg['type']:
                                    #     a0 = a
                                    # elif 'A1' == arg['type']:
                                    #     a1 = a
                                for item_sub in sent:
                                    if item_sub['id'] not in all_id and item_sub['parent'] == p[0] \
                                            and item_sub['pos'] not in filter_part:
                                        if item_sub['relate'] == 'SBV':
                                            SBV.append(item_sub['id'])
                                            all_id.append(item_sub['id'])
                                        elif item_sub['relate'] == 'VOB':
                                            VOB.append(item_sub['id'])
                                            all_id.append(item_sub['id'])
                                        # elif item_sub['relate'] == 'ADV':
                                        #     p_ADV.append(item_sub['id'])
                                        #     all_id.append(item_sub['id'])
                                if len(a0) + len(SBV) != 0 and len(a1) + len(VOB) != 0:
                                    # print(a0, '#', p, '#', a1, '#', date)
                                    # event = '@'.join(list(filter(lambda x:x!='',[a0, p, a1])))
                                    # event = '@'.join([a0, p, a1])
                                    # print(event)
                                    # remove_digits = str.maketrans('', '', digits)
                                    # event = '@'.join([event.translate(remove_digits), date, title])
                                    for item_sub in sent:
                                        # print(item_sub['cont'])
                                        if item_sub['id'] not in all_id and item_sub['relate'] == 'ATT' \
                                                and item_sub['pos'] not in filter_part:
                                            if item_sub['parent'] in a0 + SBV:
                                                a0_com.append(item_sub['id'])
                                                all_id.append(item_sub['id'])
                                            elif item_sub['parent'] in a1 + VOB:
                                                a1_com.append(item_sub['id'])
                                                all_id.append(item_sub['id'])

                                    # print("SBV:", SBV)
                                    # print("a0_com:", a0_com)
                                    # print("a0:", a0)
                                    # print("p_ADV:", p_ADV)
                                    # print("p:", p)
                                    # print("a1_com:", a1_com)
                                    # print("a1:", a1)
                                    # print("VOB:", VOB)
                                    event_subject = []
                                    event_object = []
                                    event_verb = []
                                    for id in filter(lambda x: x != '', SBV + a0_com + a0 + p_ADV):
                                        event_subject.append(sent[id]['cont'])
                                    for id in filter(lambda x: x != '', a1_com + a1 + VOB):
                                        event_object.append(sent[id]['cont'])
                                    event_verb.append(sent[p[0]]['cont'])
                                    # print("event_subject:", list(filter(lambda x: x != '', SBV + a0_com + a0)))
                                    # print("event_verb:", event_verb)
                                    # print("event_object:", list(filter(lambda x: x != '', a1_com + a1 + VOB)))

                                    event_subject = ' '.join(event_subject)
                                    event_verb = ' '.join(event_verb)
                                    event_object = ' '.join(event_object)

                                    # event = '@'.join([event_subject, event_verb, event_object, date, title])
                                    # print("#", event)
                                    write_sub.append('@'.join([event_subject, event_verb, event_object, date]) + '\n')
                                    count += 1
                                    if count % 100 == 0:
                                        print("process %d, number %d" % (i, count))
            except Exception as e:
                print(e)
                pass

        lock.acquire()
        try:
            with open(EVENT_TRAIN, 'a', encoding="utf8") as e:
                for item in write_sub:
                    e.write(item)
            print("process %d, finish writing" % i)
        finally:
            lock.release()

    thrs = [threading.Thread(target=thread_process, args=(i, num, 0)) for i in range(num)]
    [thr.start() for thr in thrs]
    [thr.join() for thr in thrs]


if __name__ == '__main__':
    process(4)

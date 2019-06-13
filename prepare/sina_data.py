import json as js
import os

DATA_PATH = "/home/haijun/project/causality/crawl/sina-crawler/data"
RESULT_PATH = "../data/all/all_events.txt"
SINA_PATH = "/home/haijun/project/causality/crawl/sina-crawler/sina_news_5586474.json"


# from DATA_PATH
def get_hourly_data():
    files = os.listdir(DATA_PATH)
    events = []
    for file in files:
        if not os.path.isdir(file):
            with open(DATA_PATH + "/" + file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line != "":
                        try:
                            event = js.loads(line)
                            title = event["title"].replace('\n', '').replace('\t', '').strip()
                            time = event["time"].replace('\n', '').replace('\t', '').replace('年', '-').replace('月',
                                                                                                               '-').replace(
                                '日', ' ').strip()
                            if title != '' and time != '':
                                events.append([title, time])
                        except:
                            pass

    print(len(events))
    return events


# from SINA_PATH
def get_sina_data():
    events = []
    with open(SINA_PATH, 'r') as sp:
        for line in sp:
            line = line.strip()
            if line != '':
                try:
                    event = js.loads(line)
                    title = event["title"].replace('\n', '').replace('\t', '').strip()
                    time = event["time"].replace('\n', '').replace('\t', '').replace('年', '-').replace('月',
                                                                                                       '-').replace('日',
                                                                                                                    ' ').strip()
                    if title != '' and time != '':
                        events.append([title, time])
                except:
                    pass
    print(len(events))
    return events


if __name__ == '__main__':
    hourly_result = get_hourly_data()
    sina_result = get_sina_data()
    result = hourly_result + sina_result

    print("begin writing:", len(result))
    with open(RESULT_PATH, 'w') as res:
        for item in result:
            res.write('\t'.join(item))
            res.write('\n')

    print("finish")

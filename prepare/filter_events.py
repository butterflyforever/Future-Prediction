import json

from constants import stop_words_file


def get_stopwords():
    stop_words = []
    with open(stop_words_file, 'r') as f:
        for line in f:
            line = line.strip()
            stop_words.append(line)
    return stop_words


def remove_stop(raw_data, no_stop_data):
    stop_words = get_stopwords()
    print("Begin removing stop words")
    data = []
    with open(raw_data, 'r') as rd:
        for line in rd:
            line = list(filter(lambda x: x, line.strip().split('@')))
            if len(line) != 4:
                continue
            line_refine = []
            for item in line[0:3]:
                item_words = item.split(' ')
                item_words_refine = []
                for word in item_words:
                    if word not in stop_words:
                        item_words_refine.append(word)
                item_words_refine = ' '.join(item_words_refine)
                line_refine.append(item_words_refine)
            line_refine = list(filter(lambda x: x, line_refine))
            if len(line_refine) != 3:
                continue
            line_refine = '@'.join(line_refine + [line[3]])
            data.append(line_refine)
            # print(line_refine + '\n')
            # if len(a) == 100:
            #     break
    with open(no_stop_data, 'w') as ns:
        ns.write('\n'.join(data))


def get_triple_length(triple):
    event = ''.join(triple).replace(' ', '')
    return len(event)


def remove_noise(raw_data, less_noise, threshold):
    vocab_file = '../data/all/1000/Vocab_in_events_for_training.json'
    vocab = json.load(open(vocab_file, 'r'))[0]
    data = []
    short = 0
    noise = 0
    all = 0
    has_noise = 0
    with open(raw_data, 'r') as rd:
        for line in rd:
            all += 1
            PASS = True
            line_check = line.strip().split('@')
            if get_triple_length(line_check[0:3]) < 5:
                # print("Too short:", line.strip())
                short += 1
                continue
            for item in line_check[0:3]:
                Noise = 0.0
                item_words = item.split(' ')
                for word in item_words:
                    if int(vocab[word]) < 15:
                        Noise += 1
                if Noise / len(item_words) > threshold:
                    PASS = False
                    noise += 1
                    # print("No Pass:", line.strip())
                    break
                if Noise > 0:
                    has_noise += 1
            if PASS:
                data.append(line.strip())
            # if len(a) > 100:
            #     break
    print("all:",all)
    print("short:",short)
    print("noise:",noise)
    print("pass:",len(data))
    print("has noise in pass:",has_noise)
    with open(less_noise, 'w') as ln:
        ln.write('\n'.join(data))


if __name__ == '__main__':
    raw_data = "../data/all/1000/refine_all_events_15to18.txt"
    no_stop_data = "../data/all/1000/no_stop_all_events_15to18.txt"
    less_noise = "../data/all/1000/less_noise_all_events_15to18.txt"
    # remove_stop(raw_data, no_stop_data)
    remove_noise(no_stop_data, less_noise, 0.8)

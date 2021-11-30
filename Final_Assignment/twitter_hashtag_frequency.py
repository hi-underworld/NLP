# Chap02-03/twitter_hashtag_frequency.py
import sys
from collections import Counter
import json

def get_hashtags(tweet):
    entities = tweet.get('entities', {})
    hashtags = entities.get('hashtags', [])
    return [tag['text'].lower() for tag in hashtags]

if __name__ == '__main__':

    fname = 'data/TweetsNBA.json'
    MAX_LINE = 1000
    ct = 0
    with open(fname, 'r') as f:
        hashtags = Counter()
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            tweet = json.loads(line)
            hashtags_in_tweet = get_hashtags(tweet)
            hashtags.update(hashtags_in_tweet)
            if ct < MAX_LINE:
                ct += 1
            else:
                break

        for tag, count in hashtags.most_common(20):
            print("{}: {}".format(tag, count))

    # fname = '/Users/feili/Downloads/TweetsNBA.json'
    # fname_o = '/Users/feili/Downloads/TweetsNBA_1000.json'
    # MAX_LINE = 1000
    # ct = 0
    # with open(fname, 'r') as f:
    #     f_out = open(fname_o, 'w')
    #     for line in f:
    #         line = line.strip()
    #         if len(line) == 0:
    #             continue
    #         f_out.write(line+'\n')
    #         if ct < MAX_LINE:
    #             ct += 1
    #         else:
    #             break
    #
    #     f_out.close()

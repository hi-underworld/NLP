# Chap02-03/twitter_influence.py
import sys
import json

def usage():
    print("Usage:")
    print("python {} <username1> <username2>".format(sys.argv[0]))

if __name__ == '__main__':

    screen_name1 = 'trump'
    followers_file1 = 'data/{}/followers.jsonl'.format(screen_name1)
    with open(followers_file1) as f1:
        reach1 = []
        reach2 = []
        for line in f1:
            profile = json.loads(line)
            reach1.append((profile['screen_name'], profile['followers_count']))

    profile_file1 = 'data/{}/user_profile.jsonl'.format(screen_name1)

    with open(profile_file1) as f1:
        profile1 = json.load(f1)

        followers1 = profile1['followers_count']

        tweets1 = profile1['statuses_count']


    sum_reach1 = sum([x[1] for x in reach1])
    sum_reach2 = sum([x[1] for x in reach2])
    avg_followers1 = round(sum_reach1 / followers1, 2)


    timeline_file1 = 'data/trump/user_timeline_{}.jsonl'.format(screen_name1)

    with open(timeline_file1) as f1:
        favorite_count1, retweet_count1 = [], []
        favorite_count2, retweet_count2 = [], []
        for line in f1:
            tweet = json.loads(line)
            favorite_count1.append(tweet['favorite_count'])
            retweet_count1.append(tweet['retweet_count'])

    avg_favorite1 = round(sum(favorite_count1) / tweets1, 2)

    avg_retweet1 = round(sum(retweet_count1) / tweets1, 2)

    favorite_per_user1 = round(sum(favorite_count1) / followers1, 2)

    retweet_per_user1 = round(sum(retweet_count1) / followers1, 2)

    print("----- Stats {} -----".format(screen_name1))
    print("{} followers".format(followers1))
    print("{} users reached by 1-degree connections".format(sum_reach1))
    print("Average number of followers for {}'s followers: {}".format(screen_name1, avg_followers1))
    print("Favorited {} times ({} per tweet, {} per user)".format(sum(favorite_count1), avg_favorite1, favorite_per_user1))
    print("Retweeted {} times ({} per tweet, {} per user)".format(sum(retweet_count1), avg_retweet1, retweet_per_user1))

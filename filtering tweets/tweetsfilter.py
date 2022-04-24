import os
import pandas as pd



path = "/Users/pradyg/minio/data/tweets/"
stocks = os.listdir(path)
if ".DS_Store" in stocks:
    stocks.remove(".DS_Store")

def gettweets(date, stocks):
    tweets = []
    for s in stocks:
        path = "/Users/pradyg/minio/data/tweets/" + s + "/" + date + "/"
        try:
            filenames = os.listdir(path)
            for f in filenames:
                with open(path+f) as file:
                    lines = file.readlines()
                for l in lines:
                    tweets.append(l)
        except:
            pass
    return tweets

tweets = []
days = 6
for i in range(1, days):
    temp = gettweets("2022-04-0" + str(i), stocks)
    tweets += temp

len(tweets)

Emoji_description = pd.read_csv(os.getcwd() + "/support.md", delimiter="|", skiprows=12)

Emoji_description = Emoji_description.drop('Unnamed: 0', 1)
Emoji_description = Emoji_description.drop('Unnamed: 5', 1)

def get_emoji_dict(Emoji_description):
    d = {}
    l = len(list(Emoji_description.iloc[:, 0]))
    ucode = list(Emoji_description.iloc[:, 3])
    tags = list(Emoji_description.iloc[:, 2])
    names = list(Emoji_description.iloc[:, 1])

    for i in range(l):
        clean_cur_tag = str(tags[i]).replace(" ", "")
        cur_ucode = str(ucode[i])
        if clean_cur_tag != "nan" and clean_cur_tag != "":
            if ";" in tags[i]:
                d[cur_ucode.replace(" ", "")] = clean_cur_tag[:clean_cur_tag.index(";")]
            else:
                d[cur_ucode.replace(" ", "")] = clean_cur_tag
        else:
            clean_cur_tag = str(names[i]).replace(" ", "")
            d[cur_ucode.replace(" ", "")] = clean_cur_tag
    return d

d = get_emoji_dict(Emoji_description)

def getindices(s):
    ret = []
    for i in range(len(s)-3):
        if s[i:i+2] == "\\u":
            ret.append(i)
    return ret

def greedymatch(s, d):
    idx = getindices(s)
    if len(idx) == 0:
        return s
    if len(idx) > 5:
        return ''
    for i in reversed(range(len(idx))):
        if (len(s) > idx[i]+6) and (s[idx[0]:idx[i]+6] in d):
            s = s.replace(s[idx[0]:idx[i]+6], ". " +d[s[idx[0]:idx[i]+6]] + ". ")
            return greedymatch(s, d)
    s = s.replace(s[idx[0]:idx[0]+6], "")
    return greedymatch(s, d)

def clean_tweets(tweets, d):
    for i in range(len(tweets)):
        tweets[i] = greedymatch(tweets[i], d)

    for tweet in tweets:
        if tweet == '':
            tweets.remove(tweet)
    
    for i in range(len(tweets)):
        if "\\n" in tweets[i]:
            tweets[i].replace("\\n", "")
    
    return tweets

tweets = clean_tweets(tweets, d)

len(tweets)

out_file = open("tweets_data.txt", "w+")

for tweet in tweets:
    out_file.write(tweet + "\n")

out_file.close()


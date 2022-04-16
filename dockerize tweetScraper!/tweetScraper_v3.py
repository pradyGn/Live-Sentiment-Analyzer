from minio import Minio
from minio.error import (InvalidResponseError)
import tweepy
import os
import time
from datetime import date
import datetime
import json
import io
import re

def makelists(num):
    ret = []
    for n in range(num):
        ret.append([])
    return ret

class MyListener(tweepy.Stream):
    def __init__(self, time_limit=60*60*8):
        super(MyListener, self).__init__(
            consumer_key, consumer_secret,
            access_token, access_token_secret
        )
        self.start_time = time.time()
        self.limit = time_limit
        self.stock_list = getstocklist()
        self.minioclient = getMinioClient('accesskey', 'secretkey')
        self.fifteen_minute_counter = time.time() #### CHANGES MADE HERE
        self.two_hour_counter = time.time()
        self.tweets15 = makelists(10)
        self.tweets120 = makelists(10)

    def on_status(self, data):
        if (time.time() - self.start_time) < self.limit:
            try:

                #### CHANGES MADE BELOW

                if hasattr(data, 'retweeted_status') and hasattr(data.retweeted_status, 'extended_tweet'):
                    tweet = data.retweeted_status.extended_tweet['full_text']
                if hasattr(data, 'extended_tweet'):
                    tweet = data.extended_tweet['full_text']
                else:
                    tweet = data.text
                
                tweet = re.sub(r"http\S+", "", tweet)

                try:
                    for i in range(len(self.stock_list)):
                        if self.stock_list[i] in tweet:
                            self.tweets15[i].append(tweet)
                            self.tweets120[i].append(tweet)

                except (BaseException, InvalidResponseError) as e:
                    print("Error on_data 0: %s" % str(e))

                if (time.time() - self.fifteen_minute_counter) >= 15*60:
                    try :
                        for i in range(len(self.tweets15)):
                            print(self.tweets15[i])
                            tweets_as_bytes = json.dumps(self.tweets15[i], indent=2).encode('utf-8')
                            streamfile = io.BytesIO(tweets_as_bytes)
                            self.minioclient.put_object(
                            "tweets",
                            f"15_minutes/" + self.stock_list[i][1:] + "/" + str(date.today()) + "/" + str(datetime.datetime.now().time()),
                            streamfile,
                            len(tweets_as_bytes)
                            )
                        self.fifteen_minute_counter = time.time()
                        
                    except (BaseException, InvalidResponseError) as e:
                        print("Error on_data 1: %s" % str(e))

                if (time.time() - self.two_hour_counter) >= 2*60*60:
                    try :
                        for i in range(len(self.tweets120)):
                            print(self.tweets120[i])
                            tweets_as_bytes = json.dumps(self.tweets120[i], indent=2).encode('utf-8')
                            streamfile = io.BytesIO(tweets_as_bytes)
                            self.minioclient.put_object(
                            "tweets",
                            f"2_hour/" + self.stock_list[i][1:] + "/" + str(date.today()) + "/" + str(datetime.datetime.now().time()),
                            streamfile,
                            len(tweets_as_bytes)
                            )
                        self.two_hour_counter = time.time()

                    except (BaseException, InvalidResponseError) as e:
                        print("Error on_data 1: %s" % str(e))

                #### CHANGES MADE ON TOP

                return True
            except (BaseException, InvalidResponseError) as e:
                print("Error on_data 1: %s" % str(e))
            return True
        else:
            print("Returned False")
            self.running = False
            return False
 
    def on_error(self, status):
        print(status)
        return True


def getMinioClient(access, secret):
    return Minio(
        '172.18.0.1:9000',
        access_key=access,
        secret_key=secret,
        secure=False
    )

def getstocklist():
    with open('input.txt') as stocknames:
        lines = stocknames.readlines()

    for i in range(len(lines)):
        if '\n' in lines[i]:
            lines[i] = lines[i].replace('\n', '')

    return lines


if __name__ == '__main__':

    consumer_key = "a0JpDjO0BlQTphicnx4umMBig"
    consumer_secret = "e5L79vmde3o3uAmMMTZMnyRL2rzJTIkWBPpnbpoNrHXJOuwG2E"
    access_token = "1498424937110978564-ujtqXt7lpEzPG5fHNsQoQXBe10qaBl"
    access_token_secret = "ebvSphhvVHTKfUmQ1SxCuzCl3CzVWbANL6xszr711ef0f"

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    print(os.getcwd())

    stock_list = getstocklist()

    print(stock_list)

    twitter_stream = MyListener()
    twitter_stream.filter(track=stock_list)
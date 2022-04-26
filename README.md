# Live-Sentiment-Analyzer

First a docker network needs to be created for this specific project. This can be done using the following command,

```
docker network create [network_name]
```


Now, the minio container should be run on port 9000 and following command can be used to run it.

```
docker run -t -d -p 9000:9000 --name [container_name] --network [network_name] \
-e "MINIO_ACCESS_KEY" = accesskey \
-e "MINIO_SECRET_KEY" = secretkey \
-v [file_in_home_directory]:/data \
minio/minio server /data
```

Additionally, any access_key and secret_key can be used. However, one needs to update these passwords in the tweetScraper_v3.py file.



Once the miniotweets container is up and running one can dockerize the tweetscraper by using the following commands,

```
docker build -t tweet_scraper_image .
```
```
docker run -d --name tweetScraperContainer --network [network_name] tweet_scraper_container
```

.
.
.
.

Question to wonder on...

Why to collect data in a batch 15 mins and 2 hours?
- A trader wants a real time sentiment analysis of a certain stock. Collecing data in a batch of lesser than 15 minutes would result in too less of data points for certain stocks. The data points are collected in 2 hour batch as well so that sentiment over longer duration of files can be assessed with minial number of files accessed.

What about the emojis?
- Emojis have been replace with their description.
- Example: I love Tesla! 🤗 -> I love Tesla! Hugging face.

What about ads?
- I plan to build a seperate simple model to filter out ads. I'll probably start with a simple Navie Bayes and check if complex DL models are required or not.

How is the collected data being used for actual sentiment analysis? Isn't this data unlabeled?
- I have taken inspiration from the semi-supervised learning model in computer vision to solve this problem.
- Reference: https://arxiv.org/pdf/1905.00546.pdf

Which model to actully perform the heavy weight lifting (Sentiment Analysis)?
- e-MLM finBERT pre-trained and fine tuned on collected data.

.
.
.
.

The flow chat of how backend (so far) would look is as follows,

![alt text](https://github.com/pradyGn/Live-Sentiment-Analyzer/blob/main/backend_flowchart.png?raw=true)

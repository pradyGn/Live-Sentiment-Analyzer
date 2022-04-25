# Live-Sentiment-Analyzer

First a docker network needs to be created for this specific project. This can be done using the following command,

```
docker network create [network_name]
```


Now, the minio container should be run on port 9000 and following command can be used to run it.

```
docker run -t -d -p 9000:9000 --name [container_name] -- network [network_name] \
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


The flow chat of how backend would look is as follows,

![alt text](https://github.com/pradyGn/Live-Sentiment-Analyzer/blob/main/backend_flowchart.png?raw=true)

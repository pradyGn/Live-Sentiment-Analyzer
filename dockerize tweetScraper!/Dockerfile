FROM python:3.8.8

WORKDIR /tweetScraper15mins

ADD requirments.txt tweetScraper_v3.py input.txt .

RUN pip install -r requirments.txt

CMD ["python", "tweetScraper_v3.py"]
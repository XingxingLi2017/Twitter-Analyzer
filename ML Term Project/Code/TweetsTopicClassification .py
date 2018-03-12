# TWITTER SETTINGS
# Credentials to use Twitter API
TWITTER_CONSUMER_KEY = '9nubPxUd7RbeISTLb2EIS6Wn3'
TWITTER_CONSUMER_SECRET = 'u16gzuRPZwZ2MURmHGAAfbRbzU6XZEtpV2hqVhQwWF5ntvzk23'
TWITTER_ACCESS_TOKEN_KEY = '764527996950745088-KTGnxHEtb2ywTautHn9yZUwEf0O9WPW'
TWITTER_ACCESS_TOKEN_SECRET = 'ybT4B0Le9A7SXJGGjdSE5MIHvZ7NpcZrleVjou9PqHXoT'

# This is the twitter user that we will be profiling using our news classifier.
TWITTER_USER = 'nytimes'
# globle value
FRIENDS_NUMBER = 100
TWEETS_NUMBER = 100
FAVORITES_NUMBER = 100
EXPAND_TWEETS = True # turn on the expansion process to expand the tweets we get

import tweepy
import re

auth = tweepy.OAuthHandler(TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET)
auth.set_access_token(TWITTER_ACCESS_TOKEN_KEY, TWITTER_ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

from random import shuffle

def get_friends_descriptions(api, twitter_account, max_users = 100):
    user_ids = api.friends_ids(twitter_account)
    shuffle(user_ids)

    following = []  # all the information of following users
    for start in range(0, min(max_users, len(user_ids)), 100):
        end = start+100
        following.extend(api.lookup_users(user_ids[start:end]))

    descriptions = []
    for user in following:
        # eliminate the http urls in user description
        description = re.sub(r'(https?://\S+)', '' , user.description)

        # discard the descriptions that are less than 10 words
        if len(re.split(r'[^0-9A-Za-z]+', description)) > 10:
            descriptions.append(description.strip('#').strip('@'))
    return descriptions

descriptions = get_friends_descriptions(api, TWITTER_USER, max_users=FRIENDS_NUMBER)

def get_tweets(api, twitter_user, tweet_type='timeline', max_tweets=200, min_words=5):
    tweets = []     # return value, subset of tull_tweets
    full_tweets = []
    step = 200
    for start in range(0, max_tweets, step):
        count = min(step, max_tweets-start)
        kwargs = {"count": count}
        if full_tweets:
            last_id = full_tweets[-1].id
            kwargs['max_id'] = last_id - 1

        if tweet_type == 'timeline':
            current = api.user_timeline(twitter_user, **kwargs)
        else:
            current = api.favorites(twitter_user, **kwargs)

        full_tweets.extend(current)

    # grade every tweets and only hold the text of the tweets
    for tweet in full_tweets:
        text = re.sub(r'(https?://\S+)', '', tweet.text)

        score = tweet.favorite_count + tweet.retweet_count
        if tweet.in_reply_to_status_id_str:
            score -= 15

        if len(re.split(r'[^0-9A-Za-z]+', text)) > min_words:
            tweets.append((text, score))

    return tweets

tweets = []     # tweets = [(text, score)...]
tweets.extend(get_tweets(api, TWITTER_USER, 'timeline', max_tweets=TWEETS_NUMBER))
tweets.extend(get_tweets(api, TWITTER_USER, 'favorites', max_tweets=FAVORITES_NUMBER))

# sort the tweets based on scores then discard the scores
tweets = sorted(tweets, key = lambda t: t[1], reverse=True)
tweets = list(map(lambda t: t[0], tweets))[:500]    # only keep the highest 500 tweets

# detect language with MonkeyLearn API

# only remain English tweets and bios

# Key to use Monkey Learn API
MONKEYLEARN_TOKEN = '814342b905e34baeff86c96c1c71c05c7a54e928'

# This classifier is used to detect the tweet/bio's language
MONKEYLEARN_LANG_CLASSIFIER_ID = 'cl_hDDngsX8'

# This classifier is used to detect the tweet/bio's topics
MONKEYLEARN_TOPIC_CLASSIFIER_ID = 'cl_5icAVzKR'

# This extractor is used to extract keywords from tweets and bios
MONKEYLEARN_EXTRACTOR_ID = 'ex_y7BPYzNG'

text_list = ["Harvard University is a private Ivy League research university in Cambridge, \
Massachusetts, established in 1636, whose history, influence, and wealth have made it one of\
 the world's most prestigious universities."]

import json
from monkeylearn import MonkeyLearn

# monkey learn api handler
ml = MonkeyLearn(MONKEYLEARN_TOKEN)

'''classify a list of texts in batches'''
def classify_batch(text_list, classifier_id):
    results = []     # return value:
    step = 250
    for start in range(0, len(text_list),step):
        end = start + step

        data = text_list[start:end]
        response = ml.classifiers.classify(classifier_id, data, sandbox=False)  # every element in the text list will have a result
        results.extend(response.result)
    return results

def filter_language(texts, language='English'):
    # Get the language of the tweets and bios using language classifier
    lang_classifications = classify_batch(texts, MONKEYLEARN_LANG_CLASSIFIER_ID)

    # discard the non-English conotent
    lang_texts = [text for text, prediction in zip(texts, lang_classifications) if prediction[0]['label'] == language]

    return lang_texts

descriptions_english = filter_language(descriptions)
print('descriptions found: {}'.format(len(descriptions_english)))

tweets_english = filter_language(tweets)
print('Tweets found: {}'.format(len(tweets_english)))

'''--------------------------------------------------'''
### using keywords extractor to get keywords


# get key words of every text in the list
# return: [[result.class],[result.class]...]
def extract_keywords(text_list, max_keywords=10):
    results = []
    # get the result of keywords extractor
    response = ml.extractors.extract(MONKEYLEARN_EXTRACTOR_ID, text_list,max_keywords=max_keywords)
    print("extract_keywords, response length = ", len(response.result))
    results.extend(response.result)
    return results

# print(extract_keywords(tweets_english, 10))

# use Bing search engine to expand context of data to increase accuracy
BING_KEY = 'bf61452f55a2461b92eff2e4e5f94c54'


import http.client, urllib.parse, json
host = "api.cognitive.microsoft.com"
path = "/bing/v7.0/search"
term = 'Microsoft Cognitive Services'

import multiprocessing.dummy as multiprocessing

# return the top 5 names and snippets of the search string
def _bing_search(search):
    MAX_EXPANSIONS = 5
    headers = {'Ocp-Apim-Subscription-Key': BING_KEY}
    connection = http.client.HTTPSConnection(host)
    query = urllib.parse.quote(search)
    connection.request("GET", path + "?q=" + query, headers=headers)
    response = connection.getresponse()

    response = json.loads(response.read().decode('utf8'))

    texts = []
    if 'webPages' in response:
        for result in response["webPages"]['value'][:MAX_EXPANSIONS]:
            texts.append(result['name'])
            texts.append(result['snippet'])
    else:
        return u' is be add done'
    return u' '.join(texts)

# print('_bing_search result = ', _bing_search('hello world!'))

def _expand_text(text):
    result = text + u'\n' + _bing_search(text)
    # print('In _expand_text function, result of expand = ', result)
    return result


# result of query = tags+keywords
# result of expansion for one query = tags+keywords+'\n'+search result
def expand_texts(texts):
    # first extract hashtags and keywords from the text to form the query
    queries = []
    # keywords of every text in texts
    keywords_list = extract_keywords(texts)
    # print('in expand_texts function, keywords lists = ', keywords_list)
    for text, keywords in zip(texts, keywords_list):
        query = ' '.join([item['keyword'] for item in keywords])
        query = query.lower()
        # get the tag of tweets
        tags = re.findall(r"#(\w+)", text)
        for tag in tags:
            tag = tag.lower()
            if tag not in query:
                query = tag + ' ' + query
        queries.append(query)

    # print("In expand_texts function queries are:", queries)
    pool = multiprocessing.Pool(2)
    return pool.map(_expand_text, queries)


# use Bing search to expand the context of descriptions
expanded_descriptions = descriptions_english
# expanded_descriptions = expand_texts(descriptions_english)
# print('----------descriptions before expansion is :', descriptions_english)
# print('----------expanded descriptions is :', expanded_descriptions)
if EXPAND_TWEETS:
    expanded_tweets = expand_texts(tweets_english)
else:
    expanded_tweets = tweets_english

### Detect the topics with MonkeyLearn API


from collections import Counter


def category_histogram(texts, short_texts):
    # Classify the bios and tweets with MonkeyLearn's news classifier.
    topics = classify_batch(texts, MONKEYLEARN_TOPIC_CLASSIFIER_ID)

    # The histogram will keep the counters of how many texts fall in
    # a given category.
    histogram = Counter()
    samples = {}

    for classification, text, short_text in zip(topics, texts, short_texts):

        # Join the parent and child category names in one string.
        category = classification[0]['label']
        probability = classification[0]['probability']

        if len(classification) > 1:
            category += '/' + classification[1]['label']
            probability *= classification[1]['probability']

        MIN_PROB = 0.0
        # Discard texts with a predicted topic with probability lower than a treshold
        if probability < MIN_PROB:
            continue

        # Increment the category counter.
        histogram[category] += 1

        # Store the texts by category
        samples.setdefault(category, []).append((short_text, text))

    return histogram, samples


# Classify the expanded bios of the followed users using MonkeyLearn, return the historgram
descriptions_histogram, descriptions_categorized = category_histogram(expanded_descriptions, descriptions_english)

# Print the catogories sorted by most frequent
for topic, count in descriptions_histogram.most_common():
    print(count, topic)

# Classify the expanded tweets using MonkeyLearn, return the historgram
print('before category_histogram function, expanded_tweets = ', expanded_tweets)
tweets_histogram, tweets_categorized = category_histogram(expanded_tweets, tweets_english)

# Print the catogories sorted by most frequent
for topic, count in tweets_histogram.most_common():
    print(count, topic)

### Plot the most popular topics

# from IPython import get_ipython
# get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt

# Add the two histograms (bios and tweets) to a total histogram
total_histogram = tweets_histogram + descriptions_histogram

# Get the top N categories by frequency
max_categories = 6
top_categories, values = zip(*total_histogram.most_common(max_categories))

# Plot the distribution of the top categories with a pie chart
plt.figure(1, figsize=(5, 5))
ax = plt.axes([0.1, 0.1, 0.8, 0.8])

plt.pie(
    values,
    labels=top_categories,
    shadow=True,
    colors=[
        (0.86, 0.37, 0.34), (0.86, 0.76, 0.34), (0.57, 0.86, 0.34), (0.34, 0.86, 0.50),
        (0.34, 0.83, 0.86), (0.34, 0.44, 0.86), (0.63, 0.34, 0.86), (0.86, 0.34, 0.70),
    ],
    radius=20,
    autopct='%1.f%%',
)

plt.axis('equal')
plt.show()

### Get the keywords of each category with MonkeyLearn API

joined_texts = {}

for category in tweets_categorized:
    if category not in top_categories:
        continue

    expanded = 0
    joined_texts[category] = u' '.join(map(lambda t: t[expanded], tweets_categorized[category]))

keywords = dict(zip(joined_texts.keys(), extract_keywords(joined_texts.values(), 20)))

for cat, kw in keywords.iteritems():
    top_relevant = map(
        lambda x: x.get('keyword'),
        sorted(kw, key=lambda x: float(x.get('relevance')), reverse=True)
    )

    print
    u"{}: {}".format(cat, u", ".join(top_relevant))

from IPython.display import Javascript

libs = [
    "http://d3js.org/d3.v3.min.js",
    "http://www.jasondavies.com/wordcloud/d3.layout.cloud.js"
]


def plot_wordcloud(wordcloud):
    return Javascript("""
                var fill = d3.scale.category20b();

                var cloudNode = $('<div id="wordcloud"></div>');
                element.append(cloudNode);

                var wordData = JSON.parse('%s');
                console.log(wordData);

                function draw(words) {
                    d3.select("#wordcloud").append("svg")
                        .attr("width", 600)
                        .attr("height", 502)
                        .append("g")
                        .attr("transform", "translate(300,160)")
                        .selectAll("text")
                        .data(words)
                        .enter().append("text")
                        .style("font-size", function (d) { return d.size + "px"; })
                        .style("font-family", "impact")
                        .style("fill", function (d, i) { return fill(i); })
                        .attr("text-anchor", "middle")
                        .attr("transform", function (d) {
                            return "translate(" + [d.x, d.y] + ")rotate(" + d.rotate + ")";
                        })
                        .text(function (d) { return d.text; });
                }
                console.log($("#wordcloud"));

                d3.layout.cloud().size([600, 502])
                    .timeInterval(10)
                    .words(wordData)
                    .padding(1)
                    .rotate(function () { return 0; })
                    .font('impact')
                    .fontSize(function (d) { return d.size; })
                    .on("end", draw)
                    .start();

        """ % json.dumps(wordcloud), lib=libs)


wordcloud = map(
    lambda s: {'text': s['keyword'], 'size': 15 + 40 * float(s['relevance'])},
    keywords['Society/Special Occasions']
)
plot_wordcloud(wordcloud)



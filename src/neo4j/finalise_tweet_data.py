'''Fetch last Twitter data, when linking is complete'''

from .graph import Graph
from ..twitter import Twitter
import logging
from tqdm.auto import tqdm
from typing import Dict
import pandas as pd
from pathlib import Path
import os

logger = logging.getLogger(__name__)


def populate_follows():
    '''Populate all follows relations related to linked tweets'''

    # Initialise the connection to the graph database
    graph = Graph()

    # Initialise the Twitter API wrapper
    twitter = Twitter()

    # Find the list of user IDs for the authors of all the linked tweets
    logger.info('Fetching all authors of linked tweets')
    query = '''
        MATCH (t:Tweet)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)
        WHERE r.score > 0.7 AND
              NOT exists((:User)-[:FOLLOWS]-(:User)-[:POSTED]->(t))
        RETURN DISTINCT toInteger(t.tweetId) AS tweet_id
    '''
    result_df = graph.query(query)
    tweet_ids = result_df.tweet_id.tolist()

    # Batch the tweet IDs
    batches = [tweet_ids[i:i+15] for i in range(0, len(tweet_ids) - 15, 15)]

    # Set up queries
    tweet_author_query = '''
        UNWIND $tweet_ids AS tweetId
        MATCH (u:User)-[:POSTED]->(:Tweet {tweetId:toInteger(tweetId)})
        RETURN DISTINCT toInteger(u.userId) AS user_id
    '''
    follower_creation_query = '''
        UNWIND $data AS data
        MATCH (u1:User {userId:toInteger(data.author_id)})
        MERGE (u2:User {userId:toInteger(data.follower_id)})
        MERGE (u2)-[:FOLLOWS]->(u1)
    '''
    followee_creation_query = '''
        UNWIND $data AS data
        MATCH (u1:User {userId:toInteger(data.author_id)})
        MERGE (u2:User {userId:toInteger(data.followee_id)})
        MERGE (u2)<-[:FOLLOWS]-(u1)
    '''

    for batch in tqdm(batches, desc='Populating follows'):
        author_df = graph.query(tweet_author_query, tweet_ids=batch)
        if len(author_df):
            author_ids = author_df.user_id.tolist()

            # Get tweet followers of the authors
            author_followers = twitter.get_followers(author_ids)['users']
            follower_data = [dict(author_id=row.followee, follower_id=user_id)
                             for user_id, row in author_followers.iterrows()]

            # Get tweet followees of the authors
            author_followees = twitter.get_followees(author_ids)['users']
            followee_data = [dict(author_id=row.follower, followee_id=user_id)
                             for user_id, row in author_followees.iterrows()]

            # Populate the followers and followees
            graph.query(follower_creation_query, data=follower_data)
            graph.query(followee_creation_query, data=followee_data)


def populate_replies():
    '''Populate all the replies and quotes tweets to the linked tweets'''

    # Initialise the Twitter wrapper
    twitter = Twitter()

    # Initialise the connection to the graph database
    graph = Graph()

    # Find the list of tweet IDs for all the linked tweets
    logger.info('Fetching all linked tweets')
    query = '''
        MATCH (t:Tweet)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)
        WHERE NOT exists((:Reply)-[:REPLY_TO|QUOTE_OF]->(t)) AND
              r.score > 0.7 AND
              (t.quoteCount > 0 OR t.replyCount > 0)
        WITH t, t.quoteCount + t.replyCount AS num_replies
        RETURN DISTINCT toInteger(t.tweetId) AS tweet_id, num_replies
        ORDER BY num_replies DESC
    '''
    result_df = graph.query(query)
    tweet_ids = result_df.tweet_id.tolist()

    create_reply_query = '''
        UNWIND $data AS replyData
        MERGE (u:User {userId:toInteger(replyData.user_id)})
        WITH u, replyData
        MERGE (r:Reply {tweetId:toInteger(replyData.tweet_id)})
        WITH u, r
        MATCH (t:Tweet {tweetId:toInteger($tweet_id)})
        MERGE (u)-[:POSTED]->(r)-[:REPLY_TO]->(t)
    '''
    create_quote_query = '''
        UNWIND $data AS replyData
        MERGE (u:User {userId:toInteger(replyData.user_id)})
        WITH u, replyData
        MERGE (r:Reply {tweetId:toInteger(replyData.tweet_id)})
        WITH u, r
        MATCH (t:Tweet {tweetId:toInteger($tweet_id)})
        MERGE (u)-[:POSTED]->(r)-[:QUOTE_OF]->(t)
    '''

    for tweet_id in tqdm(tweet_ids, desc='Populating replies'):
        reply_dfs = twitter.query(conversation_id=tweet_id,
                                  max_results=500,
                                  progress_bar=False)

        if len(reply_dfs['tweets']):

            reply_tweet_ids = reply_dfs['tweets'].index.tolist()
            reply_user_ids = reply_dfs['tweets'].author_id.tolist()
            reply_data = [dict(tweet_id=tweet_id, user_id=int(user_id))
                           for tweet_id, user_id in zip(reply_tweet_ids,
                                                        reply_user_ids)]
            graph.query(create_reply_query,
                        data=reply_data,
                        tweet_id=tweet_id,
                        progress_bar=False)


        quote_dfs = twitter.query(f'url:{tweet_id}',
                                  max_results=500,
                                  progress_bar=False)

        if len(quote_dfs['tweets']):
            quote_tweet_ids = quote_dfs['tweets'].index.tolist()
            quote_user_ids = quote_dfs['tweets'].author_id.tolist()
            quote_data = [dict(tweet_id=tweet_id, user_id=int(user_id))
                          for tweet_id, user_id in zip(quote_tweet_ids,
                                                       quote_user_ids)]
            graph.query(create_quote_query,
                        data=quote_data,
                        tweet_id=tweet_id,
                        progress_bar=False)


def populate_retweets():
    '''Populate retweets of the linked tweets'''

    # Initialise the connection to the graph database
    graph = Graph()

    # Initialise the Twitter API wrapper
    twitter = Twitter()

    # Set up queries
    total_count_query = '''
        MATCH (t:Tweet)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)
        WHERE r.score > 0.7 AND t.retweetCount > 0
        RETURN count(DISTINCT toInteger(t.tweetId)) AS num
    '''
    remaining_count_query = '''
        MATCH (t:Tweet)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)
        WHERE r.score > 0.7 AND
              t.retweetCount > 0 AND
              NOT exists((t)<-[:RETWEETED]-(:User)) AND
              NOT exists(t.parsed)
        RETURN count(DISTINCT toInteger(t.tweetId)) AS num
    '''
    get_tweet_query = '''
        MATCH (t:Tweet)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)
        WHERE r.score > 0.7 AND
              t.retweetCount > 0 AND
              NOT exists((t)<-[:RETWEETED]-(:User)) AND
              NOT exists(t.parsed)
        WITH t
        LIMIT 1
        SET t.parsed = true
        RETURN DISTINCT toInteger(t.tweetId) AS tweet_id,
               t.retweetCount AS retweetCount
        ORDER BY retweetCount DESC
    '''
    retweet_creation_query = '''
        UNWIND $data AS data
        MATCH (t:Tweet {tweetId:toInteger(data.tweet_id)})
        MERGE (u:User {userId:toInteger(data.user_id)})
        MERGE (u)-[:RETWEETED]->(t)
    '''

    # Set up progress bar
    total = graph.query(total_count_query).num.tolist()[0]
    remaining = graph.query(remaining_count_query).num.tolist()[0]
    pbar = tqdm(total=total, desc='Populating retweets')
    pbar.update(total - remaining)

    while True:

        # Fetch a tweet with no retweets
        tweet_df = graph.query(get_tweet_query)

        # If no results are found we are done, so break out of the loop
        if len(tweet_df) == 0:
            break

        # Get the tweet ID
        tweet_id = tweet_df.tweet_id.tolist()[0]

        # Get the users that retweeted the tweets
        retweet_users = twitter.get_retweets(tweet_id)['users']
        retweet_data = [dict(tweet_id=int(row.retweeted), user_id=int(idx))
                        for idx, row in retweet_users.iterrows()]

        # Populate the retweet users
        graph.query(retweet_creation_query, data=retweet_data)

        # Update progress bar
        old_remaining = int(remaining)
        remaining = graph.query(remaining_count_query).num.tolist()[0]
        pbar.update(old_remaining - remaining)


def populate_timelines():
    '''Populate timelines of the users'''

    # Initialise the connection to the graph database
    graph = Graph()

    # Initialise the Twitter API wrapper
    twitter = Twitter()

    # Set up queries
    total_count_query1 = '''
        MATCH (t:Tweet)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)-[:HAS_LABEL]->(:Label)
        WHERE r.score > 0.7
        WITH DISTINCT t
        MATCH (u:User)-[:POSTED|RETWEETED]->(t)
        return count(DISTINCT u) AS num
    '''
    total_count_query2 = '''
        MATCH (t:Tweet)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)-[:HAS_LABEL]->(:Label)
        WHERE r.score > 0.7
        WITH DISTINCT t
        MATCH (u:User)-[:POSTED]->(:Reply)-[:REPLY_TO|QUOTE_OF]->(t)
        return count(DISTINCT u) AS num
    '''
    total_count_query3 = '''
        MATCH (t:Tweet)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)-[:HAS_LABEL]->(:Label)
        WHERE r.score > 0.7
        WITH DISTINCT t
        MATCH (u:User)-[:FOLLOWS]-(:User)-[:POSTED]->(t)
        return count(DISTINCT u) AS num
    '''
    remaining_count_query1 = '''
        MATCH (t:Tweet)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)-[:HAS_LABEL]->(:Label)
        WHERE r.score > 0.7
        WITH t
        MATCH (u:User)-[:POSTED|RETWEETED]->(t)
        WHERE NOT exists(u.parsed) AND NOT exists((u)-[:POSTED]->(:Tweet))
        RETURN count(DISTINCT u) AS num
    '''
    remaining_count_query2 = '''
        MATCH (t:Tweet)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)-[:HAS_LABEL]->(:Label)
        WHERE r.score > 0.7
        WITH t
        MATCH (u:User)-[:POSTED]->(:Reply)-[:REPLY_TO|QUOTE_OF]->(t)
        WHERE NOT exists(u.parsed) AND NOT exists((u)-[:POSTED]->(:Tweet))
        RETURN count(DISTINCT u) AS num
    '''
    remaining_count_query3 = '''
        MATCH (t:Tweet)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)-[:HAS_LABEL]->(:Label)
        WHERE r.score > 0.7
        WITH DISTINCT t
        MATCH (u:User)-[:FOLLOWS]-(:User)-[:POSTED]->(t)
        WHERE NOT exists(u.parsed) AND NOT exists((u)-[:POSTED]->(:Tweet))
        return count(DISTINCT u) AS num
    '''
    get_user_query1 = '''
        MATCH (t:Tweet)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)-[:HAS_LABEL]->(:Label)
        WHERE r.score > 0.7
        WITH t
        MATCH (u:User)-[:POSTED|RETWEETED]->(t)
        WHERE NOT exists(u.parsed) AND NOT exists((u)-[:POSTED]->(:Tweet))
        WITH u
        LIMIT 100
        SET u.parsed = true
        RETURN u.userId AS user_id
    '''
    get_user_query2 = '''
        MATCH (t:Tweet)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)-[:HAS_LABEL]->(:Label)
        WHERE r.score > 0.7
        WITH t
        MATCH (u:User)-[:POSTED]->(:Reply)-[:REPLY_TO|QUOTE_OF]->(t)
        WHERE NOT exists(u.parsed) AND NOT exists((u)-[:POSTED]->(:Tweet))
        WITH u
        LIMIT 100
        SET u.parsed = true
        RETURN u.userId AS user_id
    '''
    get_user_query3 = '''
        MATCH (t:Tweet)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)-[:HAS_LABEL]->(:Label)
        WHERE r.score > 0.7
        WITH t
        MATCH (u:User)-[:FOLLOWS]-(:User)-[:POSTED]->(t)
        WHERE NOT exists(u.parsed) AND NOT exists((u)-[:POSTED]->(:Tweet))
        WITH u
        LIMIT 100
        SET u.parsed = true
        RETURN u.userId AS user_id
    '''
    timeline_creation_query = '''
        UNWIND $data AS data
        MERGE (t:Tweet {tweetId:toInteger(data.tweet_id)})
        WITH t, data
        MATCH (u:User {userId:toInteger(data.user_id)})
        MERGE (u)-[:POSTED]->(t)
    '''

    # Set up progress bar
    total = (graph.query(total_count_query1).num.tolist()[0] +
             graph.query(total_count_query2).num.tolist()[0] +
             graph.query(total_count_query3).num.tolist()[0])
    remaining = (graph.query(remaining_count_query1).num.tolist()[0] +
                 graph.query(remaining_count_query2).num.tolist()[0] +
                 graph.query(remaining_count_query3).num.tolist()[0])
    pbar = tqdm(total=total, desc='Populating timelines')
    pbar.update(total - remaining)

    user_type = -1
    finished = {0: False, 1: False, 2: False}
    while True:
        user_type = (user_type + 1) % 3

        # Fetch a user
        if 1 + user_type == 1:
            user_df = graph.query(get_user_query1)
        elif 1 + user_type == 2:
            user_df = graph.query(get_user_query2)
        else:
            user_df = graph.query(get_user_query3)

        # If no results are found we are done, so break out of the loop
        if len(user_df) == 0:
            finished[user_type] = True
            if set(list(finished.values())) == {True}:
                break
            else:
                continue

        # Get the user IDs
        user_ids = user_df.user_id.tolist()

        # Get the latest 10 timeline tweets of the user
        timeline_data = list()
        for user_id in user_ids:
            try:
                timeline_tweets = twitter.get_timeline(user_id)['tweets']
                data = [dict(tweet_id=int(idx), user_id=int(row.author))
                        for idx, row in timeline_tweets.iterrows()]
                timeline_data.extend(data)
            except RuntimeError:
                continue

        # Populate the retweet users
        graph.query(timeline_creation_query, data=timeline_data)

        # Update progress bar
        old_remaining = int(remaining)
        remaining = (graph.query(remaining_count_query1).num.tolist()[0] +
                     graph.query(remaining_count_query2).num.tolist()[0] +
                     graph.query(remaining_count_query3).num.tolist()[0])
        pbar.update(old_remaining - remaining)

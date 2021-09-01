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


def fetch_followers():
    '''Fetch all the followers of the users in the database'''

    # Initialise path
    dump_dir = Path('/media/secure/dan/db_dump')
    target_path = dump_dir / 'followers.csv'

    # Initialise the connection to the graph database
    graph = Graph()

    # Initialise the Twitter API wrapper
    twitter = Twitter()

    # Find the list of user IDs for the authors of all the linked tweets
    logger.info('Fetching all authors of linked tweets')
    query = '''
        MATCH (u:User)-[:POSTED]->(t:Tweet)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)
        WHERE r.score > 0.7
        RETURN DISTINCT u.userId as user_id
    '''
    result_df = graph.query(query)
    user_ids = result_df.user_id.tolist()

    if target_path.exists():
        existing_user_ids = pd.read_csv(target_path).id.tolist()
        user_ids = [user_id for user_id in user_ids
                    if user_id not in existing_user_ids]

    user_id_batches = [user_ids[i:i+15]
                       for i in range(0, len(user_ids) - 15, 15)]

    user_df = pd.DataFrame()
    for user_id_batch in tqdm(user_id_batches, desc='Fetching followers'):
        twitter_dfs = twitter.get_followers(user_id_batch)
        user_df = user_df.append(twitter_dfs['users'])

        # Save user df
        user_df.to_csv(target_path)


def fetch_followees():
    '''Fetch all the followers of the users in the database'''

    # Initialise path
    dump_dir = Path('/media/secure/dan/db_dump')
    target_path = dump_dir / 'followees.csv'

    # Initialise the connection to the graph database
    graph = Graph()

    # Initialise the Twitter API wrapper
    twitter = Twitter()

    # Find the list of user IDs for the authors of all the linked tweets
    logger.info('Fetching all authors of linked tweets')
    query = '''
        MATCH (u:User)-[:POSTED]->(t:Tweet)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)
        WHERE r.score > 0.7
        RETURN DISTINCT u.userId as user_id
    '''
    result_df = graph.query(query)
    user_ids = result_df.user_id.tolist()

    if target_path.exists():
        existing_user_ids = pd.read_csv(target_path).id.tolist()
        user_ids = [user_id for user_id in user_ids
                    if user_id not in existing_user_ids]

    user_id_batches = [user_ids[i:i+15]
                       for i in range(0, len(user_ids) - 15, 15)]

    user_df = pd.DataFrame()
    for user_id_batch in tqdm(user_id_batches, desc='Fetching followees'):
        twitter_dfs = twitter.get_followees(user_id_batch)
        user_df = user_df.append(twitter_dfs['users'])

        # Save user df
        user_df.to_csv(target_path)


def fetch_retweets():
    '''Fetch all the replies to the linked tweets in the database'''

    # Initialise the connection to the graph database
    graph = Graph()

    # Find the list of conversation IDs for all the linked tweets
    logger.info('Fetching all linked tweets')
    query = '''
        MATCH (c:Conversation)<-[:IS_PART_OF]-(t:Tweet)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)
        WHERE r.score > 0.7
        RETURN DISTINCT c.conversationId AS conversation_id
    '''
    result_df = graph.query(query)
    conversation_ids = result_df.conversation_id.tolist()

    dump_dir = Path('/media/secure/dan/db_dump')
    source_path = dump_dir / 'source_tweets.txt'
    target_path = dump_dir / 'retweets.jsonl'

    if not source_path.exists():
        source_path.touch()

    with source_path.open('w') as f:
        f.write('\n'.join(map(str, conversation_ids)))

    os.system(f'twarc retweets {source_path} > {target_path}')

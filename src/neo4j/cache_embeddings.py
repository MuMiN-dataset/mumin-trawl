'''Cache parts of the Neo4j database, to reduce memory usage'''

from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
from .graph import Graph


CACHE_DIR = Path('/media/secure/dan/neo4j_cache')


def cache_tweet_embeddings(overwrite: bool = False):

    # Ensure that cache directory exists
    if not CACHE_DIR.exists():
        CACHE_DIR.mkdir()

    # Initialse path
    path = CACHE_DIR / 'tweet_embeddings.csv'

    # Remove csv file if it already exists and `overwrite` is true; otherwise
    # raise an error
    if path.exists():
        if overwrite:
            path.unlink()
        else:
            raise RuntimeError(f'The file {path} already exists!')

    # Initialise graph
    graph = Graph()

    while True:
        # Fetch embeddings
        query = '''
            MATCH (t:Tweet)
            WHERE exists(t.embedding)
            RETURN t.tweetId as tweet_id, t.embedding as embedding
            LIMIT 100
        '''
        embedding_df = graph.query(query).set_index('tweet_id')

        # Store embeddings; set header=0 if and only if file does not already
        # exist
        with path.open('a') as f:
            embedding_df.to_csv(f, header=(f.tell() == 0))

        # Remove embeddings from graph
        query = '''
            UNWIND $tweet_ids as tweetId
            MATCH (t:Tweet {tweetId:tweetId})
            REMOVE t.embedding
        '''
        graph.query(query, tweet_ids=embedding_df.index.tolist())

        print('Cached 100 embeddings!')

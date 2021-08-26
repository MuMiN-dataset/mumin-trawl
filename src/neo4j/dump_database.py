'''Make a dump of the database, for public release'''

import pandas as pd
from pathlib import Path
import logging
from tqdm.auto import tqdm
from .graph import Graph


logger = logging.getLogger(__name__)


def dump_cosine_similarities():
    '''Dumps the cosine similarities for further analysis'''
    # Initialise paths
    dump_dir = Path('/media') / 'secure' / 'dan' / 'db_dump'
    path = dump_dir / 'similarities.csv'

    # Define cosine similarity query
    query = '''
        MATCH (:Tweet)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)
        RETURN r.score AS similarity
        ORDER BY similarity DESC
    '''

    # Initialise graph database connection
    graph = Graph()

    # Get the data from the graph database, as a Pandas DataFrame object
    similarity_df = graph.query(query)

    # Dump the similarities
    similarity_df.to_csv(path, index=False)


def dump_database(overwrite: bool = False):
    '''Dumps the database for public release.

    Args:
        overwrite (bool, optional):
            Whether to overwrite all files if they already exist. Defaults to
            False.
    '''
    # Define node queries
    claim_query = '''
        MATCH (n:Claim)<-[r:SIMILAR]-()
        WHERE r.score > 0.7
        WITH n
        MATCH (:Reviewer)-[r:HAS_REVIEWED]->(n)
        WHERE r.predicted_verdict IN ['misinformation', 'factual']
        WITH n, collect(DISTINCT r.predicted_verdict) AS verdicts
        WHERE size(verdicts) = 1
        MATCH (rev:Reviewer)-[r:HAS_REVIEWED]->(n)
        WHERE r.predicted_verdict IN ['misinformation', 'factual']
        RETURN id(n) AS id,
               n.date AS date,
               n.language AS language,
               r.raw_verdict AS raw_verdict,
               r.predicted_verdict AS predicted_verdict,
               rev.url AS reviewer
    '''
    tweet_query = '''
        MATCH (n:Tweet)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)
        WHERE r.score > 0.7
        RETURN id(n) AS id, n.tweetId AS tweet_id
    '''
    user_query = '''
        MATCH (n:User)-[:POSTED|MENTIONS]-(:Tweet)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)
        WHERE r.score > 0.7
        RETURN id(n) AS id, n.userId AS user_id
    '''
    article_query = '''
        MATCH (n:Article)-[r:SIMILAR]-(:Claim)
        WHERE r.score > 0.7
        RETURN id(n) AS id, n.url AS url
    '''

    # Define relation queries
    user_posted_tweet_query = '''
        MATCH (src:User)-[:POSTED]->(tgt:Tweet)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)
        WHERE r.score > 0.7
        RETURN id(src) AS src, id(tgt) AS tgt
    '''
    tweet_mentions_user_query = '''
        MATCH (tgt:User)<-[:MENTIONS]-(src:Tweet)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)
        WHERE r.score > 0.7
        RETURN id(src) AS src, id(tgt) AS tgt
    '''
    user_mentions_user_query = '''
        MATCH (tgt:User)<-[:MENTIONS]-(src:User)-[:POSTED]->(:Tweet)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)
        WHERE r.score > 0.7
        RETURN id(src) AS src, id(tgt) AS tgt
    '''
    tweet_discusses_claim_query = '''
        MATCH (src:Tweet)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(tgt:Claim)
        WHERE r.score > 0.7
        RETURN id(src) AS src, id(tgt) AS tgt, r.score AS relevance
    '''
    article_discusses_claim_query = '''
        MATCH (src:Article)-[r:SIMILAR]->(tgt:Claim)
        WHERE r.score > 0.7
        RETURN id(src) AS src, id(tgt) AS tgt, r.score AS relevance
    '''

    # Dump nodes
    nodes = dict(claim=claim_query,
                 tweet=tweet_query,
                 user=user_query,
                 article=article_query)
    for name, query in tqdm(nodes.items(), desc='Dumping nodes'):
        dump_node(name, query, overwrite=overwrite)

    # Dump relations
    relations = dict(user_posted_tweet=user_posted_tweet_query,
                     tweet_mentions_user=tweet_mentions_user_query,
                     user_mentions_user=user_mentions_user_query,
                     tweet_discusses_claim=tweet_discusses_claim_query,
                     article_discusses_claim=article_discusses_claim_query)
    for name, query in tqdm(relations.items(), desc='Dumping relations'):
        dump_relation(name, query, overwrite=overwrite)


def dump_node(name: str, node_query: str, overwrite: bool = False):
    '''Dumps a node to a csv file.

    The csv file will have an `id` column for the node ID and a column for each
    node feature in `features. The file will be saved to
    `data/dump/<name>.csv`.

    Args:
        name (str):
            The name of the node, which will be the file name of the dump.
        node_query (str):
            The Cypher query that returns the node in question. The result of
            the query must return an `id` column for the node IDs, and all
            other columns will be saved as node features.
        overwrite (bool, optional):
            Whether to overwrite file if it already exists. Defaults to False.

    Raises:
        RuntimeError:
            If there is no `id` column in the resulting dataframe.
    '''
    # Initialise paths
    dump_dir = Path('/media') / 'secure' / 'dan' / 'db_dump'
    path = dump_dir / f'{name}.csv'

    # Only continue if file does not already exist, or if `overwrite` is True
    if not path.exists() or overwrite:

        # Ensure that the `data/dump` directory exists
        if not dump_dir.exists():
            dump_dir.mkdir()

        # Initialise graph database connection
        graph = Graph()

        # Get the data from the graph database, as a Pandas DataFrame object
        logger.info(f'Fetching node data on "{name}"')
        df = graph.query(node_query)

        # Drop duplicate rows
        try:
            df.drop_duplicates(inplace=True)
        except TypeError as e:
            logger.info('Dropping of duplicates failed. The error was {e}.')

        # Raise exception if `id` column is not found
        if 'id' not in df.columns:
            raise RuntimeError('There is no `id` column in the dataframe.')

        # Save the dataframe
        logger.info(f'Saving node data on "{name}"')
        df.to_csv(dump_dir / f'{name}.csv', index=False)


def dump_relation(name: str, relation_query: str, overwrite: bool = False):
    '''Dumps a relation to a csv file.

    The csv file will have a `source` column for the source node ID, a `target`
    column for the target node ID, such that two IDs on the same row are
    related by `relation`. Further columns will be relation features. The file
    will be saved to `data/dump/<name>.csv`.

    Args:
        name (str):
            The name of the relation, which will be the file name of the dump.
        relation_query (str):
            The Cypher query that returns the node in question. The result of
            the query must return a `src` column for the node ID of the source
            node and a `tgt` column for the node ID of the target column.
        overwrite (bool, optional):
            Whether to overwrite file if it already exists. Defaults to False.

    Raises:
        RuntimeError:
            If the columns of the resulting dataframe do not contain `src` and
            `tgt`.
    '''
    # Initialise paths
    dump_dir = Path('/media') / 'secure' / 'dan' / 'db_dump'
    path = dump_dir / f'{name}.csv'

    # Only continue if file does not already exist, or if `overwrite` is True
    if not path.exists() or overwrite:

        # Ensure that the `data/dump` directory exists
        if not dump_dir.exists():
            dump_dir.mkdir()

        # Initialise graph database connection
        graph = Graph()

        # Get the data from the graph database, as a Pandas DataFrame object
        logger.info(f'Fetching relation data on "{name}"')
        df = graph.query(relation_query)

        # Drop duplicate rows
        df.drop_duplicates(inplace=True)

        # Raise exception if the columns are not exactly `src` and `tgt`
        if 'src' not in df.columns or 'tgt' not in df.columns:
            raise RuntimeError(f'The dataframe for the relation "{name}" does '
                               f'not contain the columns `src` and `tgt`.')

        # Save the dataframe
        logger.info(f'Saving relation data on "{name}"')
        df.to_csv(path, index=False)

'''Make a dump of the database, for public release'''

from pathlib import Path
import logging
import io
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from typing import Dict, Union, List
import warnings
import zipfile
from .graph import Graph
from ..utils import UpdateableZipFile


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


def dump_database(overwrite: bool = False,
                  dump_nodes: bool = True,
                  dump_relations: bool = True):
    '''Dumps the database for public release.

    Args:
        overwrite (bool, optional):
            Whether to overwrite all files if they already exist. Defaults to
            False.
        dump_nodes (bool, optional):
            Whether to dump nodes in the graph. Defaults to True.
        dump_relations (bool, optional):
            Whether to dump relations in the graph. Defaults to True.
    '''
    # Remove database file if `overwrite` is True
    if overwrite:
        zip_file = Path('/media/secure/dan/db_dump/mumin.zip')
        if zip_file.exists():
            zip_file.unlink()

    # Define node queries
    claim_query = '''
        MATCH (rev:Reviewer)-[r:HAS_REVIEWED]->(n:Claim)-[:HAS_LABEL]->(l:Label)
        WHERE r.predicted_verdict IN ['misinformation', 'factual']
        WITH DISTINCT n,
             collect(DISTINCT rev.url) AS reviewers,
             collect(l.verdict)[0] AS label
        OPTIONAL MATCH (n)-[:HAS_LABEL]->(l:Label {size:'small'})
        WITH n,
             label,
             reviewers,
             l.train_mask AS small_train_mask,
             l.val_mask AS small_val_mask,
             l.test_mask AS small_test_mask
        OPTIONAL MATCH (n)-[:HAS_LABEL]->(l:Label {size:'medium'})
        WITH n,
             label,
             reviewers,
             small_train_mask,
             small_val_mask,
             small_test_mask,
             l.train_mask AS medium_train_mask,
             l.val_mask AS medium_val_mask,
             l.test_mask AS medium_test_mask
        OPTIONAL MATCH (n)-[:HAS_LABEL]->(l:Label {size:'large'})
        WITH n,
             label,
             reviewers,
             small_train_mask,
             small_val_mask,
             small_test_mask,
             medium_train_mask,
             medium_val_mask,
             medium_test_mask,
             l.train_mask AS large_train_mask,
             l.val_mask AS large_val_mask,
             l.test_mask AS large_test_mask
        RETURN id(n) AS id,
               n.keywords AS keywords,
               n.cluster_keywords AS cluster_keywords,
               n.cluster AS cluster,
               n.date AS date,
               n.language AS language,
               n.embedding AS embedding,
               label,
               reviewers,
               small_train_mask,
               small_val_mask,
               small_test_mask,
               medium_train_mask,
               medium_val_mask,
               medium_test_mask,
               large_train_mask,
               large_val_mask,
               large_test_mask
    '''
    tweet_query1 = '''
        MATCH (n:Tweet)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)-[:HAS_LABEL]->(:Label)
        WHERE r.score > 0.7
        WITH n, r.score AS relevance
        MATCH (n)-[:HAS_HASHTAG]->(:Hashtag)<-[:HAS_HASHTAG]-(t:Tweet)
        RETURN DISTINCT t.tweetId AS tweet_id, relevance
    '''
    tweet_query2 = '''
        MATCH (n:Tweet)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)-[:HAS_LABEL]->(:Label)
        WHERE r.score > 0.7
        WITH n, r.score AS relevance
        MATCH (t:Tweet)<-[:POSTED]-(:User)-[:POSTED|RETWEETED]->(n)
        RETURN DISTINCT t.tweetId AS tweet_id, relevance
    '''
    tweet_query3 = '''
        MATCH (n:Tweet)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)-[:HAS_LABEL]->(:Label)
        WHERE r.score > 0.7
        WITH n, r.score AS relevance
        MATCH (t:Tweet)<-[:POSTED]-(:User)<-[:FOLLOWS]-(:User)-[:POSTED]->(n)
        RETURN DISTINCT t.tweetId AS tweet_id, relevance
    '''
    tweet_query4 = '''
        MATCH (n:Tweet)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)-[:HAS_LABEL]->(:Label)
        WHERE r.score > 0.7
        WITH n, r.score AS relevance
        MATCH (t:Tweet)<-[:POSTED]-(:User)-[:FOLLOWS]->(:User)-[:POSTED]->(n)
        RETURN DISTINCT t.tweetId AS tweet_id, relevance
    '''
    tweet_query5 = '''
        MATCH (n:Tweet)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)-[:HAS_LABEL]->(:Label)
        WHERE r.score > 0.7
        WITH n, r.score AS relevance
        MATCH (t:Tweet)<-[:POSTED]-(u:User)-[:POSTED]->(:Reply)-[:REPLY_TO|QUOTE_OF]->(n)
        RETURN DISTINCT t.tweetId AS tweet_id, relevance
    '''
    reply_query = '''
        MATCH (t:Tweet)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)-[:HAS_LABEL]->(:Label)
        WHERE r.score > 0.7
        WITH t, r.score AS relevance
        MATCH (n:Reply)-[:REPLY_TO|QUOTE_OF]->(t)
        WHERE NOT exists((n)<-[:POSTED]-(:User)-[:RETWEETED]->(t))
        RETURN n.tweetId AS tweet_id, relevance
    '''
    user_query1 = '''
        MATCH (t:Tweet)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)-[:HAS_LABEL]->(:Label)
        WHERE r.score > 0.7
        WITH t, r.score AS relevance
        MATCH (u:User)-[:RETWEETED]->(t)
        RETURN DISTINCT u.userId AS user_id, relevance
    '''
    user_query2 = '''
        MATCH (t:Tweet)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)-[:HAS_LABEL]->(:Label)
        WHERE r.score > 0.7
        WITH t, r.score AS relevance
        MATCH (u:User)-[:FOLLOWS]-(:User)-[:POSTED]->(t)
        RETURN DISTINCT u.userId AS user_id, relevance
    '''
    # user_query3 = '''
    #     MATCH (t:Tweet)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)-[:HAS_LABEL]->(:Label)
    #     WHERE r.score > 0.7
    #     WITH t
    #     MATCH (u:User)-[:POSTED]-(:Reply)-[:REPLY_TO|QUOTE_OF]->(t)
    #     RETURN DISTINCT u.userId AS user_id
    # '''
    article_query = '''
        MATCH (n:Article)-[r:SIMILAR]->(:Claim)-[:HAS_LABEL]->(:Label)
        WHERE r.score > 0.7
        RETURN id(n) AS id, n.url AS url, r.score AS relevance
    '''

    # Define relation queries
    # user_posted_tweet_query1 = '''
    #     MATCH (n:Tweet)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)-[:HAS_LABEL]->(:Label)
    #     WHERE r.score > 0.7
    #     WITH DISTINCT n
    #     MATCH (tgt:Tweet)<-[:POSTED]-(src:User)-[:POSTED|RETWEETED]->(n)
    #     RETURN src.userId AS src, tgt.tweetId AS tgt
    # '''
    # user_posted_tweet_query2 = '''
    #     MATCH (n:Tweet)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)-[:HAS_LABEL]->(:Label)
    #     WHERE r.score > 0.7
    #     WITH DISTINCT n
    #     MATCH (tgt:Tweet)<-[:POSTED]-(src:User)-[:FOLLOWS]-(:User)-[:POSTED]->(n)
    #     RETURN src.userId AS src, tgt.tweetId AS tgt
    # '''
    # user_posted_tweet_query3 = '''
    #     MATCH (n:Tweet)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)-[:HAS_LABEL]->(:Label)
    #     WHERE r.score > 0.7
    #     WITH DISTINCT n
    #     MATCH (tgt:Tweet)<-[:POSTED]-(src:User)-[:POSTED]->(:Reply)-[:REPLY_TO|QUOTE_OF]->(n)
    #     RETURN src.userId AS src, tgt.tweetId AS tgt
    # '''
    # user_posted_tweet_query4 = '''
    #     MATCH (n:Tweet)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)-[:HAS_LABEL]->(:Label)
    #     WHERE r.score > 0.7
    #     WITH DISTINCT n
    #     MATCH (n)-[:HAS_HASHTAG]->(:Hashtag)<-[:HAS_HASHTAG]-(tgt:Tweet)<-[:POSTED]-(src:User)
    #     RETURN src.userId AS src, tgt.tweetId AS tgt
    # '''
    # user_posted_reply_query = '''
    #     MATCH (t:Tweet)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)-[:HAS_LABEL]->(:Label)
    #     WHERE r.score > 0.7
    #     WITH DISTINCT t
    #     MATCH (src:User)-[:POSTED]->(tgt:Reply)-[:REPLY_TO|QUOTE_OF]->(t)
    #     RETURN src.userId AS src, tgt.tweetId AS tgt
    # '''
    user_retweeted_tweet_query = '''
        MATCH (tgt:Tweet)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)-[:HAS_LABEL]->(:Label)
        WHERE r.score > 0.7
        WITH DISTINCT tgt, r.score AS relevance
        MATCH (src:User)-[:RETWEETED]->(tgt)
        RETURN src.userId AS src, tgt.tweetId AS tgt, relevance
    '''
    user_follows_user_query1 = '''
        MATCH (t:Tweet)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)-[:HAS_LABEL]->(:Label)
        WHERE r.score > 0.7
        WITH DISTINCT t, r.score AS relevance
        MATCH (src:User)-[r:FOLLOWS]->(tgt:User)-[:POSTED]->(t)
        RETURN src.userId AS src, tgt.userId AS tgt, relevance
    '''
    user_follows_user_query2 = '''
        MATCH (t:Tweet)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)-[:HAS_LABEL]->(:Label)
        WHERE r.score > 0.7
        WITH DISTINCT t, r.score AS relevance
        MATCH (tgt:User)<-[r:FOLLOWS]-(src:User)-[:POSTED]->(t)
        RETURN src.userId AS src, tgt.userId AS tgt, relevance
    '''
    reply_reply_to_tweet_query = '''
        MATCH (tgt:Tweet)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)-[:HAS_LABEL]->(:Label)
        WHERE r.score > 0.7
        WITH DISTINCT tgt, r.score AS relevance
        MATCH (src:Reply)-[:REPLY_TO]->(tgt)
        WHERE NOT exists((src)<-[:POSTED]-(:User)-[:RETWEETED]->(tgt))
        RETURN src.tweetId AS src, tgt.tweetId AS tgt, relevance
    '''
    reply_quote_of_tweet_query = '''
        MATCH (tgt:Tweet)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)-[:HAS_LABEL]->(:Label)
        WHERE r.score > 0.7
        WITH DISTINCT tgt, r.score AS relevance
        MATCH (src:Reply)-[:QUOTE_OF]->(tgt)
        WHERE NOT exists((src)<-[:POSTED]-(:User)-[:RETWEETED]->(tgt))
        RETURN src.tweetId AS src, tgt.tweetId AS tgt, relevance
    '''
    # tweet_mentions_user_query1 = '''
    #     MATCH (src:Tweet)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)-[:HAS_LABEL]->(:Label)
    #     WHERE r.score > 0.7
    #     WITH DISTINCT src
    #     MATCH (tgt:User)<-[:MENTIONS]-(src)
    #     RETURN src.tweetId AS src, tgt.userId AS tgt
    # '''
    # tweet_mentions_user_query2 = '''
    #     MATCH (n:Tweet)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)-[:HAS_LABEL]->(:Label)
    #     WHERE r.score > 0.7
    #     WITH DISTINCT n
    #     MATCH (tgt:User)<-[:MENTIONS]-(src:Tweet)<-[:POSTED]-(:User)-[:POSTED|RETWEETED]->(n)
    #     RETURN src.tweetId AS src, tgt.userId AS tgt
    # '''
    # tweet_mentions_user_query3 = '''
    #     MATCH (n:Tweet)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)-[:HAS_LABEL]->(:Label)
    #     WHERE r.score > 0.7
    #     WITH DISTINCT n
    #     MATCH (tgt:User)<-[:MENTIONS]-(src:Tweet)<-[:POSTED]-(:User)-[:FOLLOWS]-(:User)-[:POSTED]->(n)
    #     RETURN src.tweetId AS src, tgt.userId AS tgt
    # '''
    # tweet_mentions_user_query4 = '''
    #     MATCH (n:Tweet)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)-[:HAS_LABEL]->(:Label)
    #     WHERE r.score > 0.7
    #     WITH DISTINCT n
    #     MATCH (tgt:User)<-[:MENTIONS]-(src:Tweet)<-[:POSTED]-(:User)-[:POSTED]->(:Reply)-[:REPLY_TO|QUOTE_OF]->(n)
    #     RETURN src.tweetId AS src, tgt.userId AS tgt
    # '''
    # user_mentions_user_query1 = '''
    #     MATCH (t:Tweet)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)-[:HAS_LABEL]->(:Label)
    #     WHERE r.score > 0.7
    #     WITH DISTINCT t
    #     MATCH (src:User)-[r:MENTIONS]->(tgt:User)-[:POSTED]->(t)
    #     RETURN src.userId AS src, tgt.userId AS tgt
    # '''
    # user_mentions_user_query2 = '''
    #     MATCH (t:Tweet)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)-[:HAS_LABEL]->(:Label)
    #     WHERE r.score > 0.7
    #     WITH DISTINCT t
    #     MATCH (tgt:User)<-[r:MENTIONS]-(src:User)-[:POSTED]->(t)
    #     RETURN src.userId AS src, tgt.userId AS tgt
    # '''
    tweet_discusses_claim_query = '''
        MATCH (src:Tweet)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(tgt:Claim)-[:HAS_LABEL]->(:Label)
        WHERE r.score > 0.7
        RETURN src.tweetId AS src, id(tgt) AS tgt, r.score AS relevance
    '''
    article_discusses_claim_query = '''
        MATCH (src:Article)-[r:SIMILAR]->(tgt:Claim)-[:HAS_LABEL]->(:Label)
        WHERE r.score > 0.7
        RETURN id(src) AS src, id(tgt) AS tgt, r.score AS relevance
    '''

    # Define data types
    claim_types = dict(keywords='str',
                       cluster_keywords='category',
                       cluster='category',
                       date={'date': 'datetime64[ns]'},
                       language='category',
                       embedding='numpy',
                       label='category',
                       reviewers='O',
                       small_train_mask='bool',
                       small_val_mask='bool',
                       small_test_mask='bool',
                       medium_train_mask='bool',
                       medium_val_mask='bool',
                       medium_test_mask='bool',
                       large_train_mask='bool',
                       large_val_mask='bool',
                       large_test_mask='bool')
    tweet_types = dict(tweet_id='int64')
    reply_types = dict(tweet_id='int64')
    user_types = dict(user_id='int64')
    article_types = dict(id='int64', url='str')

    # Dump nodes
    nodes = dict(tweet=([tweet_query1,
                         tweet_query2,
                         tweet_query3,
                         tweet_query4,
                         tweet_query5], tweet_types),
                 claim=(claim_query, claim_types),
                 reply=(reply_query, reply_types),
                 user=([user_query1, user_query2], user_types),
                 article=(article_query, article_types))
    if dump_nodes:
        for name, (query, types) in nodes.items():
            dump_node(name, query, types)

    # Dump relations
    relations = dict(user_follows_user=[user_follows_user_query1,
                                        user_follows_user_query2],
                     user_retweeted_tweet=user_retweeted_tweet_query,
                     reply_reply_to_tweet=reply_reply_to_tweet_query,
                     reply_quote_of_tweet=reply_quote_of_tweet_query,
                     tweet_discusses_claim=tweet_discusses_claim_query,
                     article_discusses_claim=article_discusses_claim_query)
    if dump_relations:
        for name, query in relations.items():
            dump_relation(name, query)


def dump_node(name: str,
              node_query: Union[str, List[str]],
              data_types: Dict[str, str]):
    '''Dumps a node to a file.

    The file will be saved to `/media/secure/dan/db_dump/mumin.zip`.

    Args:
        name (str):
            The name of the node, which will be the file name of the dump.
        node_query (str or list of str):
            The Cypher query that returns the node in question. Can also be
            list of such queries, in which case the deduplicated concatenated
            results from all of them are dumped. Each query should return the
            same columns.
        data_types (dict):
            A dictionary with keys all the columns of the resulting dataframe
            and values the datatypes. Here 'numpy' is reserved for converting
            the given column into a numpy ndarray.
    '''
    # Initialise paths
    dump_dir = Path('/media') / 'secure' / 'dan' / 'db_dump'
    path = dump_dir / f'mumin.zip'

    # Define `node_queries`, a list of queries
    if isinstance(node_query, str):
        node_queries = [node_query]
    else:
        node_queries = node_query

    # Ensure that the `data/dump` directory exists
    if not dump_dir.exists():
        dump_dir.mkdir()

    # Initialise graph database connection
    graph = Graph()

    # Get the data from the graph database, as a Pandas DataFrame object
    logger.info(f'Fetching node data on "{name}"')
    df = pd.concat([graph.query(q) for q in node_queries], axis=0)

    if len(df):

        # Replace the relevance with the maximal relevance, in case of
        # duplicates
        if 'relevance' in df.columns:
            maxs = (df[[df.columns[0], 'relevance']]
                    .groupby(by=df.columns[0])
                    .max())
            df.relevance = (df.iloc[:, 0]
                              .map(lambda idx: maxs.relevance.loc[idx]))

        # Drop duplicate rows
        try:
            df.drop_duplicates(subset=df.columns[0], inplace=True)
        except TypeError as e:
            logger.info(f'Dropping of duplicates failed. The error was {e}.')

        # Convert the data types in the dataframe
        data_types_no_numpy = {col: typ for col, typ in data_types.items()
                               if typ != 'numpy'}
        df = df.astype(data_types_no_numpy)
        for col, typ in data_types.items():
            if typ == 'numpy':
                df[col] = df[col].map(lambda x: np.asarray(x))

        # Save the dataframe
        logger.info(f'Fetched {len(df)} {name} nodes. Saving node data')
        with UpdateableZipFile(path,
                               mode='a',
                               compression=zipfile.ZIP_DEFLATED) as zip_file:
            buffer = io.BytesIO()
            df.to_pickle(buffer, compression='xz', protocol=4)
            zip_file.writestr(name, buffer.getvalue())


def dump_relation(name: str, relation_query: str):
    '''Dumps a relation to a file.

    The dataframe will have a `src` column for the source node ID, a `tgt`
    column for the target node ID, such that two IDs on the same row are
    related by the relation . Further columns will be relation features. The
    file will be saved to `/media/secure/dan/db_dump/mumin.zip`.

    Args:
        name (str):
            The name of the relation, which will be the file name of the dump.
        relation_query (str):
            The Cypher query that returns the node in question. The result of
            the query must return a `src` column for the node ID of the source
            node and a `tgt` column for the node ID of the target column.

    Raises:
        RuntimeError:
            If the columns of the resulting dataframe do not contain `src` and
            `tgt`.
    '''
    # Initialise paths
    dump_dir = Path('/media') / 'secure' / 'dan' / 'db_dump'
    path = dump_dir / 'mumin.zip'

    # Define `relation_queries`, a list of queries
    if isinstance(relation_query, str):
        relation_queries = [relation_query]
    else:
        relation_queries = relation_query

    # Ensure that the `data/dump` directory exists
    if not dump_dir.exists():
        dump_dir.mkdir()

    # Initialise graph database connection
    graph = Graph()

    # Get the data from the graph database, as a Pandas DataFrame object
    logger.info(f'Fetching relation data on "{name}"')
    df = pd.concat([graph.query(q) for q in relation_queries], axis=0)

    if len(df):

        # Replace the relevance with the maximal relevance, in case of
        # duplicates
        if 'relevance' in df.columns:
            df['src_tgt'] = [f'{src}_{tgt}'
                             for src, tgt in zip(df.src, df.tgt)]
            maxs = (df[['src_tgt', 'relevance']]
                    .groupby(by='src_tgt')
                    .max())
            df.relevance = (df.src_tgt
                              .map(lambda idx: maxs.relevance.loc[idx]))
            df.drop(columns='src_tgt', inplace=True)

        # Drop duplicate rows
        try:
            df.drop_duplicates(subset=['src', 'tgt'], inplace=True)
        except TypeError as e:
            logger.info(f'Dropping of duplicates failed. The error was {e}.')

        # Raise exception if the columns are not exactly `src` and `tgt`
        if 'src' not in df.columns or 'tgt' not in df.columns:
            raise RuntimeError(f'The dataframe for the relation "{name}" '
                               f'does not contain the columns `src` and '
                               f'`tgt`.')

        # Save the dataframe
        logger.info(f'Fetched {len(df)} {name} relation. Saving relation data')
        with UpdateableZipFile(path,
                               mode='a',
                               compression=zipfile.ZIP_DEFLATED) as zip_file:
            buffer = io.BytesIO()
            df.to_pickle(buffer, compression='xz', protocol=4)
            zip_file.writestr(name, buffer.getvalue())


def hongbo_dump():
    '''Creates a database dump for use by Hongbo'''

    # Initialise dump directory
    data_dir = Path('/media') / 'secure' / 'dan' / 'db_dump' / 'hongbo_dump'
    if not data_dir.exists():
        data_dir.mkdir()

    # Initialise graph
    graph = Graph()

    # Find the top10 claim clusters with respect to number of related users
    query = '''
        MATCH (:Reviewer)-[r1:HAS_REVIEWED]->(c:Claim)<-[r2:SIMILAR|SIMILAR_VIA_ARTICLE]-(t:Tweet)
        WHERE r1.predicted_verdict = 'misinformation' AND
              r2.score > 0.7 AND
              c.cluster >= 0
        WITH c, t
        OPTIONAL MATCH (u1:User)-[:POSTED]->(t)
        WITH c, t, u1
        OPTIONAL MATCH (u2:User)<-[:MENTIONS]-(t)
        WITH c, t, u1, u2
        OPTIONAL MATCH (u3:User)<-[:MENTIONS]-(u1)
        WITH c, t, u1, u2, u3
        OPTIONAL MATCH (u4:User)-[:MENTIONS]->(u1)
        WITH c, t, u1, u2, u3, u4
        OPTIONAL MATCH (u5:User)-[:POSTED]->(:Tweet)-[:MENTIONS]->(u1)
        WITH c, t, u1, u2, u3, u4, u5
        RETURN DISTINCT c.cluster AS cluster,
               count(u1) + count(u2) + count(u3) +
               count(u4) + count(u5) AS numUsers
        ORDER BY numUsers DESC
        LIMIT 10
    '''
    top_clusters = graph.query(query).cluster.tolist()

    for cluster in tqdm(top_clusters):

        # Initialise cluster directory
        cluster_dir = data_dir / f'cluster{cluster}'
        if not cluster_dir.exists():
            cluster_dir.mkdir()

        # Get the user IDs of the users spreading misinformation related to the
        # given cluster
        query = '''
            MATCH (:Label {verdict:'misinformation'})<-[:HAS_LABEL]-(c:Claim)
            WHERE c.cluster = $cluster
            WITH DISTINCT c
            MATCH (c)<-[r:SIMILAR|SIMILAR_VIA_ARTICLE]-(t:Tweet)
            WHERE r.score > 0.7
            WITH c, t
            MATCH (u1:User)-[:POSTED]->(t)
            RETURN DISTINCT u1.userId AS user_id
        '''
        pos_df = graph.query(query, cluster=cluster)
        positive_users = pos_df.user_id.tolist()

        # Get the relations between the users
        query_out = '''
            UNWIND $user_ids AS userId
            MATCH (u1:User {userId:userId})-[:POSTED]->(t:Tweet)
            OPTIONAL MATCH (u1)-[:MENTIONS]->(u2:User)
            WITH t, u1, u2
            OPTIONAL MATCH (t)-[:MENTIONS]->(u3:User)
            WITH u1.userId AS user1_id,
                 collect(u2.userId) + collect(u3.userId) AS users
            UNWIND users AS user2_id
            RETURN user1_id, user2_id
        '''
        query_in = '''
            UNWIND $user_ids AS userId
            MATCH (u1:User {userId:userId})
            WITH u1
            OPTIONAL MATCH (u2:User)-[:MENTIONS]->(u1)
            WITH u1, u2
            OPTIONAL MATCH (u3:User)-[:POSTED]->(:Tweet)-[:MENTIONS]->(u1)
            WITH collect(u2.userId) + collect(u3.userId) AS users,
                 u1.userId AS user2_id
            UNWIND users AS user1_id
            RETURN user1_id, user2_id
        '''
        out_df = graph.query(query_out, user_ids=positive_users)
        in_df = graph.query(query_in, user_ids=positive_users)
        rel_df = (pd.concat((out_df, in_df), axis=0)
                    .drop_duplicates()
                    .reset_index(drop=True))

        # Collect the features of all the relevant users
        query = '''
            UNWIND $user_ids AS userId
            MATCH (u:User {userId:userId})
            RETURN DISTINCT u.userId AS user_id,
                   u.followingCount AS following_count,
                   u.followersCount AS followers_count,
                   u.tweetCount AS tweet_count
        '''
        cluster_users = (pd.concat((rel_df.user1_id, rel_df.user2_id), axis=0)
                           .drop_duplicates()
                           .tolist())
        feat_df = graph.query(query, user_ids=cluster_users)

        # Store the dataframes
        feat_df.to_csv(cluster_dir / 'feature.csv', index=False)
        rel_df.to_csv(cluster_dir / 'graph.csv', index=False)
        pos_df.to_csv(cluster_dir / 'pos_node.csv', index=False)

    # Initialise cluster directory
    unrelated_dir = data_dir / f'unrelated'
    if not unrelated_dir.exists():
        unrelated_dir.mkdir()

    # Collect mentions relations not related to the top clusters
    query = '''
        MATCH (u1:User)-[:POSTED]->(t:Tweet)
        WHERE exists(t.text)
        WITH t, u1
        OPTIONAL MATCH (u1)-[:MENTIONS]->(u2:User)
        WITH t, u1, u2
        OPTIONAL MATCH (t)-[:MENTIONS]->(u3:User)
        WITH u1.userId AS user1_id,
             collect(u2.userId) + collect(u3.userId) AS users
        UNWIND users AS user2_id
        RETURN user1_id, user2_id
    '''
    rel_df = graph.query(query).drop_duplicates().reset_index(drop=True)
    rel_df = rel_df[~rel_df.user1_id.isin(feat_df.user_id.tolist())]
    rel_df = rel_df[~rel_df.user2_id.isin(feat_df.user_id.tolist())]

    # Collect the unrelated users
    query = '''
        UNWIND $user_ids AS userId
        MATCH (u:User {userId:userId})
        RETURN DISTINCT u.userId AS user_id,
               u.followingCount AS following_count,
               u.followersCount AS followers_count,
               u.tweetCount AS tweet_count
    '''
    unrelated_users = (pd.concat((rel_df.user1_id, rel_df.user2_id), axis=0)
                         .drop_duplicates()
                         .tolist())
    feat_df = graph.query(query, user_ids=unrelated_users)

    # Store the dataframe with the unrelated users
    feat_df.to_csv(unrelated_dir / 'feature.csv', index=False)
    rel_df.to_csv(unrelated_dir / 'graph.csv', index=False)

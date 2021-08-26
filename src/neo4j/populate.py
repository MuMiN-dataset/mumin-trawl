'''Populate the graph database'''

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
import datetime as dt
from typing import List, Optional
from newspaper import Article, ArticleException
from tqdm.auto import tqdm
import json
import re
from neo4j.exceptions import DatabaseError
from concurrent.futures import ProcessPoolExecutor
from timeout_decorator import timeout, TimeoutError

from ..utils import root_dir, strip_url
from .graph import Graph


def populate(start_from_scratch: bool = False,
             queries: Optional[List[str]] = None):
    '''Populate the graph database with nodes and relations'''
    # Initialise the graph database
    graph = Graph()

    # Set up twitter directory and a list of all the twitter queries
    twitter_dir = Path('/media') / 'secure' / 'dan' / 'twitter'
    if queries is None:
        queries = [p for p in twitter_dir.iterdir()]

    # Delete all nodes and constraints in database
    if start_from_scratch:
        graph.query('CALL apoc.periodic.iterate('
                    '"MATCH (n) RETURN n",'
                    '"DETACH DELETE n",'
                    '{batchsize:10000, parallel:false})')

        constraints = graph.query('CALL db.constraints')
        for constraint in constraints.name:
            graph.query(f'DROP CONSTRAINT {constraint}')

    # Set up cypher directory
    cypher_dir = root_dir / 'src' / 'neo4j' / 'cypher'
    constraint_paths = list(cypher_dir.glob('constraint_*.cql'))
    node_paths = list(cypher_dir.glob('node_*.cql'))
    rel_paths = list(cypher_dir.glob('rel_*.cql'))

    # Create constraints
    for path in tqdm(constraint_paths, desc='Creating constraints'):
        cypher = path.read_text()
        graph.query(cypher)

    def load_records(query: str, fname: str) -> pd.DataFrame:
        '''Helper function to load records from a CSV file'''
        try:
            df = pd.read_csv(twitter_dir / query / f'{fname}.csv',
                             engine='python',
                             error_bad_lines=False,
                             warn_bad_lines=False)
            df = df.replace({np.nan: None})
            return df
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    def extract_urls(json_like_string: str) -> List[str]:
        '''Helper function to extract the URLs from a json-like string'''
        json_string = (json_like_string.replace('\'', '"')
                                       .replace('\\', '\\\\'))
        try:
            list_of_url_data = json.loads(json_string)
            if not isinstance(list_of_url_data, list):
                return []
            list_of_urls = list()
            for url in list_of_url_data:
                if 'expanded_url' in url:
                    url = url.get('expanded_url')
                    if url is not None:
                        list_of_urls.append(url)
                elif 'url' in url:
                    url = url.get('url')
                    if url is not None:
                        list_of_urls.append(url)
            return list_of_urls
        except json.JSONDecodeError:
            return []

    def filter_dataframes(dataframes: dict, min_retweets: int = 5) -> dict:
        '''Filter the dataframes to only contain the popular tweets'''
        # Filter the tweet dataframe
        tweets = dataframes['tweets']
        rt_count = 'public_metrics.retweet_count'
        tweets = tweets[tweets[rt_count].map(lambda x: isinstance(x, int))]
        tweets = tweets[tweets[rt_count] >= 5]
        dataframes['tweets'] = tweets

        # Filter the user dataframe
        user_ids = tweets.author_id.dropna().unique().tolist()
        if 'entities.mentions' in tweets.columns and \
                'id' in dataframes['users'].columns:
            mentions = [json.loads(m.replace('\'', '"').replace('\\', '\\\\'))
                        for m in tweets['entities.mentions']
                        if m is not None]
            user_ids.extend([m['id'] for m_list in mentions
                                     for m in m_list
                                     if 'id' in m.keys()])
        users = dataframes['users'].query('id in @user_ids')
        dataframes['users'] = users

        # Filter the place dataframes
        if 'geo.place_id' in tweets.columns and \
                'id' in dataframes['places'].columns:
            place_ids = tweets['geo.place_id'].dropna().unique().tolist()
            places = dataframes['places'].query('id in @place_ids')
            dataframes['places'] = places

        # Filter the media dataframe
        if 'attachments.media_keys' in tweets.columns and \
                'media_key' in dataframes['media']:
            media_id_lists = tweets['attachments.media_keys'].dropna().unique()
            media_id_lists = [json.loads(m.replace('\'', '"')
                                          .replace('\\', '\\\\'))
                              for m in media_id_lists]
            media_ids = list({media_id for media_id_list in media_id_lists
                                       for media_id in media_id_list})
            media = dataframes['media'].query('media_key in @media_ids')
            dataframes['media'] = media

        # Filter the poll dataframe
        if 'attachments.poll_ids' in tweets.columns and \
                'id' in dataframes['polls'].columns:
            poll_id_lists = tweets['attachments.poll_ids'].dropna().unique()
            poll_ids = list({poll_id for poll_id_list in poll_id_lists
                                       for poll_id in poll_id_list})
            polls = dataframes['polls'].query('id in @poll_ids')
            dataframes['polls'] = polls

        # Filter the replies dataframe
        if len(dataframes['replies']) > 0 and \
                'conversation_id' in dataframes['replies']:
            conversation_ids = tweets.conversation_id.dropna().unique().tolist()
            conversation_query = 'conversation_id in @conversation_ids'
            replies = dataframes['replies'].query(conversation_query)
            dataframes['replies'] = replies

        return dataframes

    def extract_all_urls(list_of_records: List[dict], col: str) -> List[str]:
        '''Helper function to extract the URLs from users and tweets'''
        all_urls = list()
        for record in list_of_records:
            if record.get(col) is not None:
                urls = extract_urls(record[col])
                all_urls.append(dict(idx=record['id'], urls=urls))
        return all_urls

    # Create nodes and relations, one query at a time
    total = (len(node_paths) + len(rel_paths)) * len(queries)
    with tqdm(total=total) as pbar:
        for query in queries:

            # Load in all the CSVs for the given query
            dfs = dict(tweets=load_records(query, 'tweets'),
                       users=load_records(query, 'authors'),
                       media=load_records(query, 'media'),
                       places=load_records(query, 'places'),
                       replies=load_records(query, 'replies'),
                       polls=load_records(query, 'polls'))

            # Filter out all the Twitter data not concerning a popular tweet
            dfs = filter_dataframes(dfs)
            records = {key: df.to_dict('records') for key, df in dfs.items()}

            # Get list of all tweet URLs for the given query
            tweet_urls = extract_all_urls(records['tweets'], 'entities.urls')
            records['tweet_urls'] = tweet_urls

            # Get list of all user URLs for the given query
            cols = ['entities.url.urls', 'entities.description.urls']
            user_urls = list()
            for col in cols:
                user_urls.extend(extract_all_urls(records['users'], col))
            for record in records['users']:
                pic = record.get('profile_image_url')
                if pic is not None:
                    url_obj = dict(idx=record['id'], urls=[pic])
                    user_urls.append(url_obj)
            records['user_urls'] = user_urls

            # Get list of all media URLs for the given query
            media_urls = list()
            for record in records['media']:
                pic = record.get('url')
                preview = record.get('preview_image_url')
                if pic is not None:
                    url_obj = dict(idx=record['media_key'],
                                   urls=[pic])
                    media_urls.append(url_obj)
                if preview is not None:
                    url_obj = dict(idx=record['media_key'],
                                   urls=[preview])
                    media_urls.append(url_obj)

            # Get list of all URLs for the given query
            records['urls'] = tweet_urls + user_urls + media_urls

            # Create all the nodes
            for path in node_paths:
                desc = f'Creating nodes ({query} - {path.stem})'
                pbar.set_description(desc)
                cypher = path.read_text()
                try:
                    graph.query(cypher, **records)
                except Exception as e:
                    print(f'The query "{query}" failed when trying to create '
                          f'the node {path.stem}. The specific error is: {e}.')
                pbar.update(1)

            # Create all the relations
            for path in rel_paths:
                desc = f'Creating relations ({query} - {path.stem})'
                pbar.set_description(desc)
                cypher = path.read_text()
                try:
                    graph.query(cypher, **records)
                except Exception as e:
                    print(f'The query "{query}" failed when trying to create '
                          f'the relation {path.stem}. The specific error is: '
                          f'{e}.')
                pbar.update(1)

@timeout(5)
def download_article_with_timeout(article):
    article.download()
    return article


def process_url(url: str):
    '''Helper function to process a url, used when creating Article nodes.'''

    # Strip the url of GET arguments
    stripped_url = strip_url(url)

    # Initialise the graph database
    graph = Graph()

    # Define the query which tags the URL as parsed
    set_parsed_query = '''
        MATCH (url:Url {name:$url})
        SET url.parsed = true
    '''

    # Check if the Article node already exists and skip if it is the case
    query = '''
        MATCH (n:Article {url:$url})
        RETURN count(n) as numArticles
    '''
    if graph.query(query, url=stripped_url).numArticles[0] > 0:
        graph.query(set_parsed_query, url=url)
        return None

    # Initialise article
    try:
        article = Article(stripped_url)
        article = download_article_with_timeout(article)
        article.parse()
    except (ArticleException, ValueError, RuntimeError, TimeoutError):
        graph.query(set_parsed_query, url=url)
        return None

    # Extract the title and skip URL if it is empty
    title = article.title
    if title == '':
        graph.query(set_parsed_query, url=url)
        return None
    title = re.sub('\n+', '\n', title)
    title = re.sub(' +', ' ', title)
    title = title.strip()

    # Extract the content and skip URL if it is empty
    content = article.text.strip()
    if content == '':
        graph.query(set_parsed_query, url=url)
        return None
    content = re.sub('\n+', '\n', content)
    content = re.sub(' +', ' ', content)
    content = content.strip()

    # Extract the authors, the publishing date and the top image
    authors = article.authors
    if article.publish_date is not None:
        publish_date = dt.datetime.strftime(article.publish_date, '%Y-%m-%d')
    else:
        publish_date = None
    top_image = article.top_image

    try:
        # Create the corresponding Article node
        query = '''
            MERGE (n:Article {url:$url})
            ON CREATE SET
                n.title = $title,
                n.authors = $authors,
                n.publish_date = $publish_date,
                n.content = $content
        '''
        graph.query(query, url=stripped_url, title=title, authors=authors,
                    publish_date=publish_date, content=content)

        # Link the Article node to the Url node
        query = '''
            MATCH (url:Url {name:$url})
            MATCH (article:Article {url:$stripped_url})
            MERGE (url)-[:IS_ARTICLE]->(article)
        '''
        graph.query(query, url=url, stripped_url=stripped_url)

        # Create the Url node for the top image if it doesn't already
        # exist
        if top_image is not None and top_image.strip() != '':
            # Create a `Url` node for the top image
            query = '''
                MERGE (:Url {name:$top_image})
            '''
            graph.query(query, top_image=top_image)

            # Connect the Article node with the top image Url node
            query = '''
                MATCH (url:Url {name:$top_image})
                MATCH (article:Article {url:$url})
                MERGE (article)-[:HAS_TOP_IMAGE]->(url)
            '''
            graph.query(query, top_image=top_image, url=stripped_url)

    # If a database error occurs, such as if the URL is too long, then
    # skip this URL
    except DatabaseError:
        graph.query(set_parsed_query, url=url)
        return None

    # Tag the Url node as parsed
    graph.query(set_parsed_query, url=url)


def create_articles(start_from_scratch: bool = False,
                    num_workers: int = 8,
                    skip_preparse: bool = False):
    '''Create article nodes from the URL nodes present in the graph database'''
    # Initialise graph
    graph = Graph()

    if start_from_scratch:
        # Remove all articles
        graph.query('CALL apoc.periodic.iterate('
                    '"MATCH (n:Article) RETURN n",'
                    '"DETACH DELETE n",'
                    '{batchsize:10000, parallel:false})')

        # Remove `parsed` from all `Url` nodes
        query = '''MATCH (url: Url)
                   WHERE exists(url.parsed)
                   REMOVE url.parsed'''
        graph.query(query)

    # Mark certain URLs as `parsed` that are not articles
    if not skip_preparse:
        get_query = '''MATCH (url: Url)
                       WHERE not exists(url.parsed) AND
                             (url.name =~ '.*youtu[.]*be.*' OR
                              url.name =~ '.*vimeo.*' OR
                              url.name =~ '.*spotify.*' OR
                              url.name =~ '.*twitter.*' OR
                              url.name =~ '.*instagram.*' OR
                              url.name =~ '.*facebook.*' OR
                              url.name =~ '.*tiktok.*' OR
                              url.name =~ '.*gab[.]com.*' OR
                              url.name =~ 'https://t[.]me.*' OR
                              url.name =~ '.*imgur.*' OR
                              url.name =~ '.*/photo/.*' OR
                              url.name =~ '.*mp4' OR
                              url.name =~ '.*mov' OR
                              url.name =~ '.*jpg' OR
                              url.name =~ '.*jpeg' OR
                              url.name =~ '.*bmp' OR
                              url.name =~ '.*png' OR
                              url.name =~ '.*gif' OR
                              url.name =~ '.*pdf')
                       RETURN url'''
        set_query = 'SET url.parsed = true'
        query = (f'Call apoc.periodic.iterate("{get_query}", "{set_query}", '
                 f'{{batchsize:1000, parallel:false}})')
        graph.query(query)

    # Get total number of Url nodes
    query = '''MATCH (url: Url)
               WHERE NOT exists(url.parsed)
               RETURN count(url) as numUrls'''
    num_urls = graph.query(query).numUrls[0]

    # Get batch of urls and create progress bar
    if num_urls > 0:
        query = f'''MATCH (url: Url)
                    WHERE NOT exists(url.parsed)
                    RETURN url.name AS url
                    LIMIT {10 * num_workers}'''
        urls = graph.query(query).url.tolist()
        pbar = tqdm(total=num_urls, desc='Parsing Url nodes')
    else:
        urls = list()

    # Continue looping until all url nodes have been visited
    while True:

        if num_workers == 1:
            for url in urls:
                process_url(url)
                pbar.update()
        else:
            with ProcessPoolExecutor(num_workers) as pool:
                pool.map(process_url, urls, timeout=10)
            pbar.update(len(urls))

        # Get new batch of urls
        query = f'''MATCH (url: Url)
                    WHERE NOT exists(url.parsed)
                    RETURN url.name AS url
                    LIMIT {10 * num_workers}'''
        result = graph.query(query)
        if len(result) > 0:
            urls = graph.query(query).url.tolist()
        else:
            break

    # Close progress bar
    if num_urls > 0:
        pbar.close()


def create_claims(start_from_scratch: bool = False):
    '''Create claim nodes in the graph database'''
    # Initialise graph
    graph = Graph()

    # Remove all claims
    if start_from_scratch:
        graph.query('CALL apoc.periodic.iterate('
                    '"MATCH (n:Claim) RETURN n",'
                    '"DETACH DELETE n",'
                    '{batchsize:10000, parallel:false})')
        graph.query('CALL apoc.periodic.iterate('
                    '"MATCH (n:Reviewer) RETURN n",'
                    '"DETACH DELETE n",'
                    '{batchsize:10000, parallel:false})')

    # Create the cypher query that creates a `Claim` node
    create_claim_query = '''
        MERGE (c:Claim {claim: $claim})
        ON CREATE SET
            c.source = $source,
            c.date = $date,
            c.language = $language
    '''

    create_reviewer_query = 'MERGE (n:Reviewer {url: $reviewer})'

    link_reviewer_claim_query = '''
        MATCH (n:Reviewer {url: $reviewer})
        MATCH (c:Claim {claim: $claim})
        MERGE (n)-[r:HAS_REVIEWED]->(c)
        ON CREATE SET
            r.raw_verdict = $raw_verdict,
            r.raw_verdict_en = $raw_verdict_en,
            r.predicted_verdict = $predicted_verdict
    '''

    reviewer_dir = Path('data') / 'reviewers'
    desc = 'Creating Claim nodes'
    for reviewer_csv in tqdm(list(reviewer_dir.iterdir()), desc=desc):
        graph.query(create_reviewer_query, reviewer=reviewer_csv.stem)
        reviewer_df = pd.read_csv(reviewer_csv)
        for _, row in reviewer_df.iterrows():
            graph.query(create_claim_query,
                        claim=row.claim,
                        date=row.date,
                        source=row.source,
                        language=row.language)
            graph.query(link_reviewer_claim_query,
                        reviewer=row.reviewer,
                        claim=row.claim,
                        raw_verdict=row.raw_verdict,
                        raw_verdict_en=row.raw_verdict_en,
                        predicted_verdict=row.predicted_verdict)

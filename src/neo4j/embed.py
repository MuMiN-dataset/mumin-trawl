'''Embed the articles and tweets in the graph database'''

from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import pandas as pd
from typing import Callable
from .graph import Graph


def embed(get_query: str,
          set_embedding_query: str,
          count_query: str,
          start_from_scratch_query: str,
          doc_preprocessing_fn: Callable = lambda x: x,
          start_from_scratch: bool = False,
          gpu: bool = True):
    '''Abstract function used to embed various nodes in the graph database.

    Args:
        get_query (str):
            The cypher query used to get a new batch. Has to output two
            columns, 'id' and 'doc', with the 'id' uniquely identifying the
            given node, and 'doc' the string to be embedded.
        set_embedding_query (str):
            The cypher query used to set the retrieved embeddings. Must take
            'id_embeddings' as an argument, which is a list of dicts, each of
            which has an 'id' entry and an 'embedding' entry.
        count_query (str):
            The cypher query used to count the remaining nodes that have not
            been embedded yet. The column containing the count must be named
            'num'.
        start_from_scratch_query (str, optional):
            The cypher query used to remove all the relevant embeddings from
            the nodes.
        doc_preprocessing_fn (callable, optional):
            A function that preprocesses the documents fetched by `get_query`.
            Needs to be a function that inputs and outputs a Pandas DataFrame,
            both of which have columns 'id' and 'doc'. Defaults to the identity
            function.
        start_from_scratch (bool, optional):
            Whether to start from scratch or not. Defaults to False.
        gpu (bool, optional):
            Whether to use the GPU. Defaults to True.
    '''
    # Initialise the graph database
    graph = Graph()

    # Remove all embeddings from the graph
    if start_from_scratch:
        graph.query(start_from_scratch_query)

    # Load the embedding model and its tokeniser
    transformer = 'paraphrase-mpnet-base-v2'
    device = 'cuda' if gpu else 'cpu'
    model = SentenceTransformer(transformer, device=device)

    # Get the total number of articles and define a progress bar
    total = graph.query(count_query).num[0]
    pbar = tqdm(total=total, desc='Adding embeddings')

    # Continue embedding until every node has been embedded
    while graph.query(count_query).num[0] > 0:

        # Fetch new ids and docs
        df = graph.query(get_query)
        df = doc_preprocessing_fn(df)
        ids = df.id.tolist()
        docs = df.doc.tolist()

        # Embed the docs
        embeddings = [list(emb.astype(float)) for emb in model.encode(docs)]

        # Set the summary as an attribute on the nodes
        id_embeddings = [dict(id=id, embedding=embedding)
                         for id, embedding in zip(ids, embeddings)]
        graph.query(set_embedding_query, id_embeddings=id_embeddings)

        # Update the progress bar
        pbar.update(len(docs))

    # Close the progress bar
    pbar.close()


def embed_articles(start_from_scratch: bool = False, gpu: bool = True):
    '''Embed all the summaries of the articles in the graph database.

    Args:
        start_from_scratch (bool, optional):
            Whether to initially remove all the embeddings. Defaults to
            False.
        gpu (bool, optional):
            Whether to use the GPU. Defaults to True.
    '''
    get_query = '''
        MATCH (n:Article)
        WHERE exists(n.summary) AND
              n.summary IS NOT NULL AND
              NOT exists(n.embedding)
        RETURN n.url AS id, n.summary AS doc
        LIMIT 3
    '''
    set_embedding_query = '''
        UNWIND $id_embeddings AS id_embedding
        WITH id_embedding.id AS url,
             id_embedding.embedding AS embedding
        MATCH (n:Article {url:url})
        SET n.embedding = embedding
    '''
    count_query = '''
        MATCH (n:Article)
        WHERE n.summary IS NOT NULL AND NOT exists(n.embedding)
        RETURN count(n) AS num
    '''
    start_from_scratch_query = '''
        MATCH (n:Article)
        WHERE exists(n.embedding)
        REMOVE n.embedding
    '''
    def clean_article(df: pd.DataFrame) -> pd.DataFrame:
        cleaned_df = df.copy()
        cleaned_df['doc'] = cleaned_df.doc.str.lower().str.strip()
        return cleaned_df

    embed(get_query=get_query,
          set_embedding_query=set_embedding_query,
          count_query=count_query,
          start_from_scratch_query=start_from_scratch_query,
          start_from_scratch=start_from_scratch,
          doc_preprocessing_fn=clean_article,
          gpu=gpu)


def embed_claims(start_from_scratch: bool = False, gpu: bool = True):
    '''Embed all the claims in the graph database.

    Args:
        start_from_scratch (bool, optional):
            Whether to initially remove all the embeddings. Defaults to
            False.
        gpu (bool, optional):
            Whether to use the GPU. Defaults to True.
    '''
    get_query = '''
        MATCH (n:Claim)
        WHERE n.claim_en IS NOT NULL AND NOT exists(n.embedding)
        RETURN n.claim AS id, n.claim_en AS doc
        LIMIT 128
    '''
    set_embedding_query = '''
        UNWIND $id_embeddings AS id_embedding
        WITH id_embedding.id AS claim,
             id_embedding.embedding AS embedding
        MATCH (n:Claim {claim:claim})
        SET n.embedding = embedding
    '''
    count_query = '''
        MATCH (n:Claim)
        WHERE n.claim_en IS NOT NULL AND NOT exists(n.embedding)
        RETURN count(n) AS num
    '''
    start_from_scratch_query = '''
        MATCH (n:Claim)
        WHERE exists(n.embedding)
        REMOVE n.embedding
    '''
    def clean_claim(df: pd.DataFrame) -> pd.DataFrame:
        cleaned_df = df.copy()
        cleaned_df['doc'] = cleaned_df.doc.str.lower().str.strip()
        return cleaned_df

    embed(get_query=get_query,
          set_embedding_query=set_embedding_query,
          count_query=count_query,
          start_from_scratch_query=start_from_scratch_query,
          start_from_scratch=start_from_scratch,
          doc_preprocessing_fn=clean_claim,
          gpu=gpu)


def embed_tweets(start_from_scratch: bool = False, gpu: bool = True):
    '''Embed all the tweets in the graph database.

    Args:
        start_from_scratch (bool, optional):
            Whether to initially remove all the embeddings. Defaults to
            False.
        gpu (bool, optional):
            Whether to use the GPU. Defaults to True.
    '''
    get_query = '''
        MATCH (n:Tweet)
        WHERE n.text_en IS NOT NULL AND NOT exists(n.embedding)
        RETURN n.tweetId AS id, n.text_en AS doc
        LIMIT 128
    '''
    set_embedding_query = '''
        UNWIND $id_embeddings AS id_embedding
        WITH id_embedding.id AS tweetId,
             id_embedding.embedding AS embedding
        MATCH (n:Tweet {tweetId:tweetId})
        SET n.embedding = embedding
    '''
    count_query = '''
        MATCH (n:Tweet)
        WHERE n.text_en IS NOT NULL AND NOT exists(n.embedding)
        RETURN count(n) AS num
    '''
    start_from_scratch_query = '''
        MATCH (n:Tweet)
        WHERE exists(n.embedding)
        REMOVE n.embedding
    '''
    def clean_tweet(df: pd.DataFrame) -> pd.DataFrame:
        cleaned_df = df.copy()
        mention_regex = r'@[a-zA-Z0-9]*'
        url_regex = r'(http[a-zA-Z0-9.\/?&:]*|www[a-zA-Z0-9.\/?&:]]*)'
        cleaned_df['doc'] = (cleaned_df.doc
                                       .str
                                       .replace(mention_regex, ' ', regex=True)
                                       .str
                                       .replace(url_regex, ' ', regex=True)
                                       .str
                                       .replace('#', ' ', regex=True)
                                       .str
                                       .replace('-+', '-', regex=True)
                                       .str
                                       .replace(' +', ' ', regex=True)
                                       .str
                                       .lower()
                                       .str
                                       .strip())
        return cleaned_df

    embed(get_query=get_query,
          set_embedding_query=set_embedding_query,
          count_query=count_query,
          start_from_scratch_query=start_from_scratch_query,
          start_from_scratch=start_from_scratch,
          doc_preprocessing_fn=clean_tweet,
          gpu=gpu)

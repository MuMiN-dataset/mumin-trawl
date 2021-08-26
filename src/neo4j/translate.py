'''Translate all the tweets in the graph database'''

from tqdm.auto import tqdm
import time
from ..translator import GoogleTranslator
from .graph import Graph


def translate_tweets(batch_size: int):
    '''Translate all the tweets in the graph database'''

    # Initialise graph and translator
    graph = Graph()
    translator = GoogleTranslator()

    # Define query that counts how many relevant tweets there are left
    count_query = '''
        MATCH (t:Tweet)
        WHERE NOT exists(t.text_en)
        RETURN count(t) as num_tweets
    '''

    # Define query that retrieves the relevant tweets
    get_query = f'''
        MATCH (t:Tweet)
        WHERE NOT exists(t.text_en)
        RETURN t.tweetId as tweet_id, t.text as text
        LIMIT {batch_size}
    '''

    # Define query that sets the `text_en` property of the relevant tweets
    set_query = '''
        UNWIND $tweet_records as tweetRecord
        MATCH (t:Tweet {tweetId: tweetRecord.tweet_id})
        SET t.text_en = tweetRecord.text_en
    '''

    total = graph.query(count_query).num_tweets[0]
    pbar = tqdm(desc='Translating tweets', total=total)
    while graph.query(count_query).num_tweets[0] > 0:
        tweet_df = graph.query(get_query)
        tweet_df['text_en'] = translator(tweet_df.text)
        tweet_df = tweet_df.drop(columns='text')
        tweet_records = tweet_df.to_dict('records')
        graph.query(set_query, tweet_records=tweet_records)
        pbar.update(len(tweet_df))

    pbar.close()


def translate_claims(batch_size: int):
    '''Translate all the claims in the graph database'''

    # Initialise graph and translator
    graph = Graph()
    translator = GoogleTranslator()

    # Define query that counts how many claims there are left
    count_query = '''
        MATCH (c:Claim)
        WHERE NOT exists(c.claim_en)
        RETURN count(c) as num_claims
    '''

    # Define query that retrieves the claims
    get_query = f'''
        MATCH (c:Claim)
        WHERE NOT exists(c.claim_en)
        RETURN c.claim as claim
        LIMIT {batch_size}
    '''

    # Define query that sets the `claim_en` property of the claims
    set_query = '''
        UNWIND $claim_records as claimRecord
        MATCH (c:Claim {claim: claimRecord.claim})
        SET c.claim_en = claimRecord.claim_en
    '''

    total = graph.query(count_query).num_claims[0]
    pbar = tqdm(desc='Translating claims', total=total)
    while graph.query(count_query).num_claims[0] > 0:
        claim_df = graph.query(get_query)
        claim_df['claim_en'] = translator(claim_df.claim)
        claim_records = claim_df.to_dict('records')
        graph.query(set_query, claim_records=claim_records)
        pbar.update(len(claim_df))

    pbar.close()


def translate_articles(batch_size: int):
    '''Translate all the articles in the graph database'''

    # Initialise graph and translator
    graph = Graph()
    translator = GoogleTranslator()

    # Define query that counts how many articles there are left
    count_query = '''
        MATCH (n:Article)
        WHERE NOT exists(n.title_en) OR NOT exists(n.content_en)
        RETURN count(n) as num_articles
    '''

    # Define query that retrieves the articles
    get_query = f'''
        MATCH (n:Article)
        WHERE (NOT exists(n.title_en) OR NOT exists(n.content_en)) AND
              NOT exists(n.notTranslatable)
        RETURN n.url as url, n.title as title, n.content as content
        LIMIT {batch_size}
    '''

    # Define query that sets the `title_en` and `content_en` properties of the
    # articles
    set_query = '''
        UNWIND $article_records as articleRecord
        MATCH (n:Article {url: articleRecord.url})
        SET n.title_en = articleRecord.title_en,
            n.content_en = articleRecord.content_en
    '''

    # Define query that sets the `title_en` and `content_en` properties of the
    # articles
    not_translatable_query = '''
        UNWIND $urls as url
        MATCH (n:Article {url: url})
        SET n.notTranslatable = true
    '''

    total = graph.query(count_query).num_articles[0]
    pbar = tqdm(desc='Translating articles', total=total)
    while graph.query(count_query).num_articles[0] > 0:
        article_df = graph.query(get_query)
        if len(article_df) == 0:
            break
        article_df['title_en'] = translator(article_df.title)

        try:
            content_en = ['\n'.join(translator(
                          [chunk for chunk in article_content.split('\n')]))
                          for article_content in article_df.content]
            article_df['content_en'] = content_en

            article_records = article_df.to_dict('records')
            graph.query(set_query, article_records=article_records)

        except RuntimeError as e:
            if ('Request payload size exceeds the limit' in str(e) or
                    'Text too long' in str(e)):
                graph.query(not_translatable_query,
                            urls=article_df.url.tolist())
            else:
                print(f'Experienced the error "{e}". Skipping.')

        pbar.update(len(article_df))

    pbar.close()

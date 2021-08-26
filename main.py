import logging
logging.basicConfig(level=logging.WARNING)
logging.getLogger('urllib3.connection').setLevel(logging.CRITICAL)
logging.getLogger('jieba').setLevel(logging.CRITICAL)

def train():
    from src.verdict_classifier.train import train
    train()

def create_claims(start_from_scratch: bool = False):
    from src.neo4j.populate import create_claims
    create_claims(start_from_scratch=start_from_scratch)

def create_articles(start_from_scratch: bool = False,
                    num_workers: int = 14,
                    skip_preparse: bool = True):
    from src.neo4j.populate import create_articles
    create_articles(start_from_scratch=start_from_scratch,
                    num_workers=num_workers,
                    skip_preparse=skip_preparse)

def summarise_articles(start_from_scratch: bool = False):
    from src.neo4j.summarise import summarise_articles
    summarise_articles(start_from_scratch=start_from_scratch)

def embed_all(start_from_scratch: bool = False):
    from src.neo4j.embed import embed_articles, embed_claims, embed_tweets
    embed_claims(start_from_scratch=start_from_scratch, gpu=True)
    embed_tweets(start_from_scratch=start_from_scratch, gpu=True)
    embed_articles(start_from_scratch=start_from_scratch, gpu=True)

def populate(start_from_scratch: bool = False):
    from src.neo4j.populate import populate
    populate(start_from_scratch=start_from_scratch)

def fetch_tweets_from_extracted_keywords():
    from src.fetch_tweets import fetch_tweets_from_extracted_keywords
    fetch_tweets_from_extracted_keywords(
        max_results_per_query=100,
        gpu=False,
        twitter_data_dir='/media/secure/dan/twitter'
    )

def fetch_facts(num_results_per_reviewer: int = 100_000,
                translate: bool = True):
    from src.fetch_facts import fetch_facts
    fetch_facts(num_results_per_reviewer=num_results_per_reviewer,
                translate=translate)

def evaluate_verdict_classifier():
    from pathlib import Path
    from src.verdict_classifier import VerdictClassifier
    try:
        ckpt = next(Path('models').glob('*.ckpt'))
        transformer = 'sentence-transformers/paraphrase-mpnet-base-v2'
        model = VerdictClassifier.load_from_checkpoint(str(ckpt),
                                                       transformer=transformer)
        model.eval()
        verdicts = ['true', 'false', 'not sure', 'sure', 'half true',
                    'half false', 'mostly true', 'mostly false']
        print(model.predict(verdicts))
    except StopIteration:
        raise RuntimeError('No pretrained verdict classifier in `models`'
                           'directory!')

def add_predicted_verdicts():
    from src.add_predicted_verdicts import add_predicted_verdicts
    add_predicted_verdicts()

def link_tweets(start_from_scratch: bool = False):
    from src.neo4j.link_nodes import link_nodes
    link_nodes(start_from_scratch=start_from_scratch, node_type='tweet')

def link_articles(start_from_scratch: bool = False):
    from src.neo4j.link_nodes import link_nodes
    link_nodes(start_from_scratch=start_from_scratch, node_type='article')

def link_all(start_from_scratch: bool = False):
    from src.neo4j.link_nodes import link_nodes
    link_nodes(start_from_scratch=start_from_scratch, node_type='article')
    link_nodes(start_from_scratch=start_from_scratch, node_type='tweet')

def translate_tweets():
    from src.neo4j.translate import translate_tweets
    translate_tweets(batch_size=1000)

def translate_claims():
    from src.neo4j.translate import translate_claims
    translate_claims(batch_size=1000)

def translate_articles():
    from src.neo4j.translate import translate_articles
    translate_articles(batch_size=2)

def dump_database(overwrite: bool = False):
    from src.neo4j.dump_database import dump_database
    dump_database(overwrite=overwrite)

def dump_claim_embeddings(overwrite: bool = False):
    from src.neo4j.dump_database import dump_node
    query = '''
        MATCH (n:Claim)<-[r:SIMILAR]-()
        WITH DISTINCT n as n
        RETURN id(n) as id,
               n.claim_en as claim,
               n.embedding as embedding
    '''
    dump_node('claim_embeddings', query, overwrite=overwrite)

def dump_cosine_similarities(overwrite: bool = False):
    from src.neo4j.dump_database import dump_cosine_similarities
    dump_cosine_similarities()

if __name__ == '__main__':
    #dump_database(overwrite=True)
    #link_all()
    embed_all()
    #dump_cosine_similarities()

'''Fetch Twitter data related to the keywords of interest'''

from .translator import GoogleTranslator
from .twitter import Twitter
from .utils import root_dir
from tqdm.auto import tqdm
from typing import List
import logging
import yaml
from keybert import KeyBERT
import pandas as pd
import datetime as dt
from pathlib import Path
from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)


def fetch_tweets(keywords: List[str] = ['coronavirus', 'covid'],
                 translate_keywords: bool = False,
                 include_conspiracy_hashtags: bool = False,
                 max_results_per_query: int = 100_000,
                 twitter_data_dir: str  = 'data/twitter',
                 progress_bar: bool = True,
                 output_dir_prefix: str = '',
                 **kwargs):
    '''Fetch all the data from Twitter related to the keywords of interest.

    The results will be stored in the 'data/twitter' folder.

    Args:
        keywords (list of str, optional):
            List of keywords to search for. These keywords will be translated
            into all the BCP47 languages and query'd separately. Defaults to
            ['coronavirus', 'covid'].
        translate_keywords (bool, optional):
            Whether to translate the `keywords` into all BCP47 languages.
            Defaults to False.
        include_conspiracy_hashtags (bool, optional):
            Whether to include all the conspiracy hashtags, which are located
            in the 'data/conspiracy_hashtags.yaml' file. Defaults to False.
        max_results_per_query (int, optional):
            The maximal number of results to get per query. Must be at least
            10. Defaults to 100_000.
        twitter_data_dir (str, optional):
            The (relative or absolute) path to the directory containing all the
            tweets. Defaults to 'data/twitter'.
        progress_bar (bool, optional):
            Whether a progress bar should be shown or not. Defaults to
            True.
        output_dir_prefix (str, optional):
            A prefix to be prepended to the output directory name for each
            keyword, meaning the folder names will be of the form
            "<output_dir_prefix><keyword>". Defaults to an empty string.
        **kwargs:
            Keyword arguments to use in Twitter.query.
    '''
    # Initialise Twitter wrapper
    twitter = Twitter()

    # Initialise the list of queries
    query_list = list()

    # Include the keywords and all their translations
    if len(keywords) > 0:

        # Add keywords to the list of queries
        query_list.extend(keywords)

        if translate_keywords:
            # Initialise translation wrapper
            translator = GoogleTranslator()

            # Load list of BCP47 languages
            bcp47_path = root_dir / 'data' / 'bcp47.yaml'
            bcp47 = [lang for lang in yaml.safe_load(bcp47_path.read_text())
                          if isinstance(lang, str)]

            # Make list of keywords
            query_list_en = query_list.copy()
            for language_code in tqdm(bcp47, desc='Translating keywords'):
                translated_queries = translator(query_list_en,
                                                target_lang=language_code)
                if isinstance(translated_queries, str):
                    translated_queries = [translated_queries]
                query_list.extend(translated_queries)

    # Include the conspiracy hashtags
    if include_conspiracy_hashtags:

        # Load list of conspiracy hashtags
        hashtags_path = root_dir / 'data' / 'conspiracy_hashtags.yaml'
        hashtags = yaml.safe_load(hashtags_path.read_text())
        hashtags = ['#' + hashtag for hashtag in hashtags]

        # Extend the list of queries with the conspiracy hashtags
        query_list.extend(hashtags)

    # Remove duplicates
    query_list = list({q.lower() for q in query_list})

    # Ensure that only strings are in the query list
    query_list = [q for q in query_list if isinstance(q, str)]

    # Define directory with all Twitter data
    if twitter_data_dir.startswith('/'):
        twitter_data_dir = Path(twitter_data_dir)
    else:
        twitter_data_dir = root_dir / twitter_data_dir

    # Create Twitter directory if it does not already exist
    if not twitter_data_dir.exists():
        twitter_data_dir.mkdir()

    # Initialise progress bar
    if progress_bar:
        pbar = tqdm(total=len(query_list))

    # Fetch loop
    for query in query_list:

        # Update progress bar description
        if progress_bar:
            pbar.set_description(f'Fetching tweets related to "{query}"')

        # Set up query subdirectory
        query_folder_name = query.replace('#', 'hashtag_').replace(' ', '_')
        # query_folder_name = output_dir_prefix + query_folder_name
        query_dir = twitter_data_dir / query_folder_name
        new_query_folder_name = output_dir_prefix + '-' + query_folder_name
        new_query_dir = twitter_data_dir / new_query_folder_name

        # Skip to next query if its directory already exists
        if query_dir.exists():
            query_dir.rename(new_query_dir)
            # Update the progress bar
            if progress_bar:
                pbar.update(1)
            continue
        elif new_query_dir.exists():
            # Update the progress bar
            if progress_bar:
                pbar.update(1)
            continue

        query_folder_name = output_dir_prefix + '-' + query_folder_name
        query_dir = twitter_data_dir / query_folder_name

        # Fetch the Twitter data
        try:
            results = twitter.query(query, max_results=max_results_per_query,
                                    is_reply=False, shares_content=True,
                                    progress_bar=progress_bar, **kwargs)
        except Exception as e:
            print(f'Error when processing {query}: "{e}"')
            continue

        if len(results['tweets']) > 0:
            # Create the query directory
            query_dir.mkdir()

            # Save the Twitter data
            for name, df in results.items():
                path = query_dir / f'{name}.csv'
                df.to_csv(path)

        # Update the progress bar
        if progress_bar:
            pbar.update(1)


def fetch_tweets_from_extracted_keywords(
        max_results_per_query: int = 1_000,
        gpu: bool = True,
        twitter_data_dir: str = 'data/twitter'):
    '''Extract keywords from all claims and fetch tweets for each of them.

    Args:
        max_results_per_query (int, optional):
            The maximal number of results to get per query. Must be at least
            10. Defaults to 1_000.
        gpu (bool, optional):
            Whether to use the GPU for keywords extraction. Defaults to True.
        twitter_data_dir (str, optional):
            The (relative or absolute) path to the directory containing all the
            tweets. Defaults to 'data/twitter'.
    '''

    # Initialise the keyword model
    transformer = 'paraphrase-multilingual-MiniLM-L12-v2'
    sbert = SentenceTransformer(transformer, device='cuda' if gpu else 'cpu')
    kw_model = KeyBERT(sbert)
    kw_config = dict(top_n=5, keyphrase_ngram_range=(1, 2))

    # Iterate over all the fact-checking websites
    reviewers = root_dir / 'data' / 'reviewers'
    desc = 'Fetching tweets'
    for reviewer_csv in tqdm(list(reviewers.iterdir()), desc=desc):

        # Load the CSV file with all the claims for the given reviewer
        reviewer_df = pd.read_csv(reviewer_csv)

        # Remove punctuation and numbers from the claims in the dataframe
        regex = r'[,.“\"0-9]'
        reviewer_df['claim'] = (reviewer_df.claim
                                           .str
                                           .replace(regex, '', regex=True))

        # Convert date column to datetime types
        reviewer_df['date'] = pd.to_datetime(reviewer_df.date, errors='coerce')
        reviewer_df.dropna(subset=['date'], inplace=True)

        # Extract the keywords from the claims
        keywords = [{kw[0]
                     for kw in kw_model.extract_keywords(claim, **kw_config)}
                    for claim in reviewer_df.claim]

        desc = f'Fetching tweets for {str(reviewer_csv.stem)}'
        pbar = tqdm(keywords, desc=desc)
        for (_, row), kw_set in zip(reviewer_df.iterrows(), pbar):
            fetch_tweets(keywords=list(kw_set),
                         max_results_per_query=max_results_per_query,
                         from_date=row.date-dt.timedelta(days=3),
                         to_date=row.date+dt.timedelta(days=1),
                         twitter_data_dir=twitter_data_dir,
                         progress_bar=False,
                         output_dir_prefix=str(reviewer_csv.stem))

'''Build list of fact-checking websites, in all BCP47 languages'''

from .translator import GoogleTranslator
from .fact_checker import FactChecker
from .utils import root_dir
from tqdm.auto import tqdm
import logging
import yaml


logger = logging.getLogger(__name__)


def build_reviewer_list():
    '''Build list of fact-checking websites, in all BCP47 languages.

    The resulting reviewer list will be saved to 'data/reviewers.yaml'.
    '''
    # Initialisation
    translator = GoogleTranslator()
    fact_checker = FactChecker()

    # Load list of BCP47 languages
    bcp47_path = root_dir / 'data' / 'bcp47.yaml'
    bcp47 = [lang for lang in yaml.safe_load(bcp47_path.read_text())
                  if isinstance(lang, str)]

    # Build list of keywords by translating into all the BCP47 languages
    query_list = ['coronavirus', 'corona', 'covid']
    query_list_en = query_list.copy()
    for language_code in tqdm(bcp47, desc='Translating keywords'):
        logger.debug(f'Translating {query_list_en} into {language_code}...')
        query_list.extend(translator(query_list_en, target_lang=language_code))
    query_list = list(set(query_list))

    # Fetch a sample of claims and verdicts for all the keywords
    all_reviewers = list()
    for query in tqdm(query_list, desc='Fetching claims and verdicts'):
        logger.debug(f'Querying {query} for the first 1000 claims...')
        df = fact_checker(query, num_results=1000)
        if len(df) > 0:
            reviewers = df.reviewer.unique().tolist()
            all_reviewers.extend(reviewers)
    all_reviewers = list(set(all_reviewers))

    # Save the list of reviewers to the yaml file
    logger.debug('Saving the list of reviewers...')
    path = root_dir / 'data' / 'reviewers.yaml'
    with path.open('w') as f:
        yaml.dump(all_reviewers, f)

    logger.info(f'Saved {len(all_reviewers)} reviewers to data/reviewers.yaml')

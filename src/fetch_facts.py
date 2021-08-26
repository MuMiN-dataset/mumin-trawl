'''Fetch claims and verdicts from the Google Fact Check API'''

import yaml
from tqdm.auto import tqdm
import logging

from .fact_checker import FactChecker
from .build_reviewer_list import build_reviewer_list
from .utils import root_dir


logger = logging.getLogger(__name__)


def fetch_facts(num_results_per_reviewer: int = 10, translate: bool = True):
    '''Fetch claims and verdicts from the Google Fact Check API.

    Args:
        num_results_per_reviewer (int, optional):
            The number of claims to fetch from each fact checking reviewer.
            Defaults to 10.
    '''
    # Initialise the factchecker
    fact_checker = FactChecker(translate=translate)

    # Load list of all fact checking websites that we're fetching from
    reviewers_path = root_dir / 'data' / 'reviewers.yaml'
    if not reviewers_path.exists():
        build_reviewer_list()
    yaml_data = reviewers_path.read_text()
    reviewers = yaml.safe_load(yaml_data)

    # Set up reviewer directory
    reviewer_dir = root_dir / 'data' / 'reviewers'
    if not reviewer_dir.exists():
        reviewer_dir.mkdir()

    # Get list of already existing claim files
    existing_reviewers = [str(path.stem)
                          for path in reviewer_dir.glob('*.csv')]

    # Fetch claims and verdicts
    pbar = tqdm(total=len(reviewers), desc='Fetching claims and verdicts')
    for reviewer in reviewers:
        pbar.set_description(f'Fetching claims and verdicts from {reviewer}')
        if reviewer not in existing_reviewers:
            df = fact_checker(reviewer=reviewer,
                              num_results=num_results_per_reviewer)

            if len(df) == 0:
                continue

            # Drop claims with no date. We have guaranteed that this is less
            # than 10% of the claims for each fact-checking institute
            df.dropna(subset=['date'], inplace=True)

            if len(df) > 0:
                df.to_csv(reviewer_dir / f'{reviewer}.csv', index=False)
                if len(df[df.date.isnull()]) / len(df) > 0.1:
                    raise Exception(f'There are many null date values '
                                    f'for {reviewer}!')
            else:
                print(f'{reviewer} returned no results!')
        pbar.update(1)
    pbar.close()

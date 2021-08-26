'''Wrapper for the Google Fact Check API'''

import requests
import pandas as pd
import logging
from typing import Optional
from dotenv import load_dotenv; load_dotenv()
import os
import re

from .translator import GoogleTranslator


logger = logging.getLogger(__name__)


class FactChecker:
    '''A wrapper for the Google Fact Check API'''

    base_url = 'https://factchecktools.googleapis.com/v1alpha1/claims:search'

    def __init__(self, translate: bool = True):
        self.translate = translate
        self._translator = GoogleTranslator() if self.translate else None
        self._api_key = os.getenv('GOOGLE_API_KEY')
        logger.debug('FactChecker initialised!')

    def __call__(self,
                 query: Optional[str] = None,
                 reviewer: Optional[str] = None,
                 language_code: Optional[str] = None,
                 max_age_days: Optional[int] = None,
                 num_results: int = 10) -> pd.DataFrame:
        '''Search for fact checked claims.

        Args:
            query (str or None, optional):
                The query to search for. If None then `reviewer` is required.
                Defaults to None.
            reviewer (str or None, optional):
                The reviewer to search for (for instance, politifact). If None
                then 'query' is required. Defaults to None.
            language_code (str, optional):
                Filter by the given language code. Defaults to no filter.
            max_age_days (int, optional):
                The maximum number of days since the claim was made, in days.
                Defaults to no limit.
            num_results (int, optional):
                The maximum number of claims to return. Can be at most 49999.
                Defaults to 10.

        Returns:
            Pandas DataFrame:
                The search results, with columns 'claim', 'date', 'source',
                'reviewer', 'language', 'raw_verdict', and 'raw_verdict_en'.
        '''
        # Set parameters for GET request
        params = dict(query=query,
                      reviewPublisherSiteFilter=reviewer,
                      languageCode=language_code,
                      maxAgeDays=max_age_days,
                      pageSize=min(49_999, num_results),
                      key=self._api_key)

        logger.debug(f'Performing GET request with parameters {params}.')
        result = requests.get(self.base_url, params=params)
        result = result.json()
        num_results = 0 if result == {} else len(result['claims'])
        logger.debug(f'GET request returned {num_results} claims.')

        # If there are no results then return an empty dataframe
        if num_results == 0:
            return pd.DataFrame()

        # Fetch the fields from the GET request result that we need
        claims = [dict(claim=claim.get('text'),
                       date=claim.get('claimDate'),
                       source=claim.get('claimant'),
                       claim_reviews=claim.get('claimReview'))
                  for claim in result['claims']]

        # Unpack the claim reviews
        logger.debug('Unpacking the claim reviews...')
        for claim in claims:
            reviews = claim.pop('claim_reviews')

            if reviews is None or len(reviews) == 0:
                reviewer = None
                language = None
                raw_verdict = None
                review_date = None
            else:
                review = reviews[0]
                publisher = review.get('publisher')
                reviewer = None if publisher is None else publisher.get('site')
                language = review.get('languageCode')
                raw_verdict = review.get('textualRating')
                review_date = review.get('reviewDate')

            claim['reviewer'] = reviewer
            claim['language'] = language
            claim['raw_verdict'] = raw_verdict
            if claim['date'] is None:
                if review_date is not None:
                    claim['date'] = review_date
                elif reviews[0].get('url') is not None:
                    review_url = reviews[0].get('url')
                    regex = r'[0-9]{4}-[0-9]{2}-[0-9]{2}'
                    date_match = re.search(regex, review_url)
                    if date_match is not None:
                        claim['date'] = date_match[0] + 'T00:00:00Z'

            if claim['date'] is None:
                logger.debug(f'This claim has no date: "{claim}"')

        if self.translate:
            # Get all the indices of the English and non-English texts
            logger.debug('Fetching the indices of the English and non-English '
                         'claims and verdicts...')
            en_idxs = [idx for idx in range(len(claims))
                           if claims[idx]['language'] == 'en']
            non_en_idxs = [idx for idx in range(len(claims))
                               if claims[idx]['language'] != 'en']

            if non_en_idxs:
                # Get the non-English verdicts
                non_en_verdicts = [claims[idx]['raw_verdict']
                                   for idx in non_en_idxs]

                # Translate the non-English verdicts
                logger.debug('Translating the non-English verdicts...')
                translated_verdicts = self._translator(non_en_verdicts)

                # Populate `claims` with the translated verdicts
                for i, idx in enumerate(non_en_idxs):
                    claims[idx]['raw_verdict_en'] = translated_verdicts[i]

            # Populate `claims` with the (trivial) English verdicts
            for idx in en_idxs:
                claims[idx]['raw_verdict_en'] = claims[idx]['raw_verdict']

        # Convert the claims into a Pandas DataFrame
        logger.debug('Converting the data into a Pandas DataFrame...')
        claim_df = pd.DataFrame.from_records(claims)

        return claim_df

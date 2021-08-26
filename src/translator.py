'''Translator classes'''

import requests
from requests.exceptions import ConnectionError
import numpy as np
import itertools as it
from typing import Union, Sequence, List
from tqdm.auto import tqdm
import logging
from dotenv import load_dotenv; load_dotenv()
import os
import time


logger = logging.getLogger(__name__)


class GoogleTranslator:
    '''A wrapper for the Google Translation API'''
    base_url = 'https://translation.googleapis.com/language/translate/v2'

    def __init__(self):
        self.api_key = os.getenv('GOOGLE_API_KEY')

    def __call__(self,
                 text: Union[str, List[str]],
                 target_lang: str = 'en') -> Union[str, List[str]]:
        '''Translate text.

        Args:
            text (str or list of str): The text to translate.

        Returns:
            str or list of str: The translated English text.
        '''
        # Ensure that `text` is a list
        if isinstance(text, str):
            text = [text]

        logger.debug(f'Translating {len(text)} documents...')

        # Set up the POST request parameters, except the query
        data = dict(target=target_lang, format='text', key=self.api_key)

        # Split into chunks
        num_batches = (len(text) // 50) + 1
        batches = np.array_split(text, num_batches)
        logger.debug(f'Split the texts into {num_batches} batches.')

        # Translate every batch and collect all the translations
        all_translations = []
        for batch_idx, batch in enumerate(batches):
            logger.debug(f'Translating batch {batch_idx}...')

            # Update the query to the batch
            data['q'] = batch

            # Perform the POST request
            while True:
                try:
                    response = requests.post(self.base_url, data=data)
                    logger.debug(f'POST request resulted in status code '
                                 f'{response.status_code}.')
                    break
                except ConnectionError:
                    logger.debug('Connection error encountered. Trying again.')
                    continue

            # If we have reached the translation quota, then wait and try again
            while response.status_code == 403:
                logger.debug('Translation quota reached. Waiting...')
                time.sleep(10)
                response = requests.post(self.base_url, data=data)
                logger.debug(f'POST request resulted in status code '
                             f'{response.status_code}.')

            # Convert the response into a dict
            data_dict = response.json()

            # Give informative error if the request failed
            if 'data' not in data_dict and 'error' in data_dict:
                error = data_dict['error']
                status_code = error.get('code')
                message = error.get('message')
                exception_message = ''
                if status_code is not None:
                    exception_message += f'[{status_code}] '
                if message is not None:
                    exception_message += message
                raise RuntimeError(exception_message)

            # Pull out the translations
            translations = [dct['translatedText']
                            for dct in data_dict['data']['translations']]

            # Update `all_translations` with the new translations
            all_translations.extend(translations)
            logger.debug(f'Have now translated {len(all_translations)} '
                         f'documents')

        # If a string was inputted, make sure to also output a string
        if len(all_translations) == 1:
            all_translations = all_translations[0]

        # Return the list of all the translations
        return all_translations


class DeepLTranslator:
    '''A wrapper for the DeepL translation API.

    API documentation available at:
    https://www.deepl.com/docs-api/translating-text/

    Args:
        api_key (str):
            An API key for the DeepL Translation API.
        progress_bar (bool, optional):
            Whether a progress bar should be shown during translation. Defaults
            to True.
    '''
    base_url = 'https://api-free.deepl.com/v2/translate'

    def __init__(self, api_key: str, progress_bar: bool = True):
        self.api_key = api_key
        self.progress_bar = progress_bar

    def __call__(self, text: Union[str, Sequence[str]]) -> str:
        '''Translate text into English.

        Args:
            text (str or sequence of str): The text to translate.

        Returns:
            str or list of str: The translated English text.
        '''
        # Ensure that `text` is a list
        if isinstance(text, str):
            text = [text]

        # Set up the DeepL API parameters
        params = dict(auth_key=self.api_key,
                      target_lang='en',
                      split_sentences=0)

        # Split into chunks of at most 50, as this is the maximum number of
        # texts that the DeepL API can process at a time.
        num_batches = (len(text) // 50) + 1
        batches = np.array_split(text, num_batches)

        # Initialise the progress bar
        if self.progress_bar:
            pbar = tqdm(desc='Translating', total=len(text))

        # Call the DeepL API to get the translations
        translations = []
        for batch in batches:
            params['text'] = batch
            responses = [requests.get(self.base_url, params=params)]

            # If input is too large (HTTP error 414) for the DeepL API then
            # split into smaller parts and process these individually.
            # Keep splitting until the input is small enough
            if responses[0].status_code == 414:
                for i in it.count(start=1):
                    responses = []
                    minibatches = np.array_split(batch, 2 ** i)
                    for minibatch in minibatches:
                        params['text'] = minibatch
                        response = requests.get(self.base_url, params=params)
                        responses.append(response)

                    status_codes = [res.status_code for res in responses]
                    if 414 not in status_codes:
                        break

            # Collect all the translations in the batch and add them to the
            # list of all translations
            results = [response.json() for response in responses]
            translation_batch = [translation['text']
                                 for result in results
                                 for translation in result['translations']]
            translations.extend(translation_batch)

            # Update the progress bar
            if self.progress_bar:
                pbar.update(len(batch))

        # Close the progress bar
        if self.progress_bar:
            pbar.close()

        # Return the single translation if the input was a single string, and
        # otherwise return the list of all the translations
        if len(translations) == 1:
            return translations[0]
        else:
            return translations

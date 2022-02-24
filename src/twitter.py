'''Wrapper for the Twitter API'''

import requests
from typing import Optional, Union, Dict, List
import pandas as pd
import datetime as dt
import time
import logging
from tqdm.auto import tqdm
from dotenv import load_dotenv; load_dotenv()
import os


logger = logging.getLogger(__name__)


class Twitter:
    '''A wrapper for the Twitter API'''

    base_url: str = 'https://api.twitter.com/2/tweets/search/all'

    def __init__(self):
        api_key = os.getenv('TWITTER_API_KEY')
        self.headers = dict(Authorization=f'Bearer {api_key}')
        self.expansions = ['attachments.poll_ids',
                           'attachments.media_keys',
                           'author_id',
                           'entities.mentions.username',
                           'geo.place_id',
                           'in_reply_to_user_id',
                           'referenced_tweets.id',
                           'referenced_tweets.id.author_id']
        self.tweet_fields = ['attachments',
                             'author_id',
                             'conversation_id',
                             'created_at',
                             'entities',
                             'geo',
                             'id',
                             'in_reply_to_user_id',
                             'lang',
                             'public_metrics',
                             'possibly_sensitive',
                             'referenced_tweets',
                             'reply_settings',
                             'source',
                             'text',
                             'withheld']
        self.user_fields = ['created_at',
                            'description',
                            'entities',
                            'id',
                            'location',
                            'name',
                            'pinned_tweet_id',
                            'profile_image_url',
                            'protected',
                            'public_metrics',
                            'url',
                            'username',
                            'verified',
                            'withheld']
        self.media_fields = ['duration_ms',
                             'height',
                             'media_key',
                             'preview_image_url',
                             'type',
                             'url',
                             'width',
                             'public_metrics']
        self.place_fields = ['contained_within',
                             'country',
                             'country_code',
                             'full_name',
                             'geo',
                             'id',
                             'name',
                             'place_type']
        self.poll_fields = ['duration_minutes',
                            'end_datetime',
                            'id',
                            'options',
                            'voting_status']

    def query(self,
              query: str = None,
              max_results: int = 10,
              from_date: Optional[dt.datetime] = None,
              to_date: Optional[dt.datetime] = None,
              from_id: Optional[str] = None,
              to_id: Optional[str] = None,
              is_reply: Optional[bool] = None,
              shares_content: Optional[bool] = None,
              language: Optional[str] = None,
              conversation_id: Optional[Union[int, str]] = None,
              progress_bar: bool = True,
              ) -> Dict[str, pd.DataFrame]:
        '''Query Twitter for tweets.

        Args:
            query (str):
                The plaintext query, mostly containing the search keyword(s).
                The complete Twitter query can also be put in here, and simply
                ignore the rest of the arguments below.
            max_results (int, optional):
                The maximal number of results to return. Must be at least 10.
                Defaults to 10.
            from_date (datetime or None, optional):
                Return tweets from this date. If None or if set to a date
                before 2006-03-21 then it will be changed to 2006-03-21, as
                no tweets exist before that time. If set to a date after
                today's date then it will be changed to today's date. Defaults
                to None.
            to_date (datetime or None, optional):
                Return tweets until this date, not inclusive. If None then it
                will be set to today's date. If set to a date before 2006-03-21
                then it will be changed to 2006-03-21, as no tweets exist
                before then. If set to a date after today's date then it will
                be changed to today's date. If equal to or before `from_date`
                then empty dataframes will be returned. Defaults to None.
            from_id (str or None, optional):
                Return tweets greater than this ID. If set then this will
                overwrite `from_date` and `to_date`. Defaults to None.
            to_id (str or None, optional):
                Return tweets with ID less than this value. If set then this
                will overwrite `from_date` and `to_date`. Defaults to None.
            is_reply (bool or None, optional):
                Whether to return replies, where a 'reply' here means a
                retweet, quote tweet or reply to a tweet. If True then only
                replies will be returned, if False then no replies will be
                returned, and if None then all kinds of tweets will be allowed.
                Defaults to None.
            shares_content (bool or None, optional):
                Whether to return tweets that share links, images or videos. If
                True then only tweets that share content will be returned, if
                False then no tweets sharing content will be returned, and if
                None then no filtering will be done on content sharing.
                Defaults to None.
            language (str or None, optional):
                Only return tweets in the language with the given BCP 47
                language tag. See e.g.
                https://www.techonthenet.com/js/language_tags.php for a list of
                all language tags. If None then no filtering will be done on
                languages. Defaults to None.
            conversation_id (int, str or None, optional):
                Only return tweets belonging to the given conversation ID. If
                None then no filtering is done on conversations. Defaults to
                None.
            progress_bar (bool, optional):
                Whether a progress bar should be shown or not. Defaults to
                True.

        Returns:
            Dictionary of strings and Pandas dataframes:
                The keys of the dictionary are 'tweets', 'authors', 'media',
                'replies', 'polls', 'places' and 'metadata', with the values
                being the respective dataframes.
        '''
        # Ensure that `from_date` and `to_date` are not before 2006-03-21
        min_datetime = dt.datetime(2006, 3, 21, tzinfo=dt.timezone.utc)
        if (from_date is None or
                (from_date is not None and from_date < min_datetime)):
            from_date = min_datetime
        if to_date is not None and to_date < min_datetime:
            to_date = min_datetime

        # Ensure that `from_date` and `to_date` are not after today's date
        max_datetime = dt.datetime.now(dt.timezone.utc)
        if from_date is not None and from_date > max_datetime:
            from_date = max_datetime
        if to_date is not None and to_date > max_datetime:
            to_date = max_datetime

        # If `from_date` and `to_date` are equal or if `from_date` happens
        # *after* `to_date` then return empty dataframes
        if (from_date is not None and
                to_date is not None and
                from_date >= to_date):
            empty_df = pd.DataFrame()
            return dict(tweets=empty_df, authors=empty_df, media=empty_df,
                        replies=empty_df, polls=empty_df, places=empty_df,
                        metadata=empty_df)

        # Initialise `query_items`. Here "-(is:nullcast)" means that we ignore
        # ad tweets
        if query is not None:
            query_items = [query, '-(is:nullcast)']
        else:
            query_items = ['-(is:nullcast)']

        # Add language to the query
        if language is not None:
            query_items.append(f'lang:{language}')

        # Add conversation id to the query
        if conversation_id is not None:
            query_items.append(f'conversation_id:{conversation_id}')

        # Add reply filtering to the query
        if is_reply is not None:
            if is_reply:
                query_items.append('(is:reply OR is:retweet OR is:quote)')
            elif not is_reply:
                query_items.append('-(is:reply OR is:retweet OR is:quote)')

        # Add content sharing filtering to the query
        if shares_content is not None:
            if shares_content:
                query_items.append('(has:media OR has:links OR has:images)')
            elif not shares_content:
                query_items.append('-(has:media OR has:links OR has:images)')

        # Build the query
        query = ' '.join(query_items)
        logger.debug(f'Built query "{query}".')

        # Initialise parameters used in the GET requests
        params = {'query': query,
                  'expansions': ','.join(self.expansions),
                  'tweet.fields': ','.join(self.tweet_fields),
                  'user.fields': ','.join(self.user_fields),
                  'media.fields': ','.join(self.media_fields),
                  'place.fields': ','.join(self.place_fields),
                  'poll.fields': ','.join(self.poll_fields)}

        # Set start ID
        if from_id is not None:
            params['since_id'] = from_id

        # Set end ID
        if to_id is not None:
            params['until_id'] = to_id

        # ID overrules date
        if from_id is not None or to_id is not None:
            from_date = None
            to_date = None

        # Set start date
        if from_date is not None:
            start_time = from_date.strftime('%Y-%m-%dT%H:%M:%SZ')
            params['start_time'] = start_time

        # Set end date
        if to_date is not None:
            end_time = to_date.strftime('%Y-%m-%dT%H:%M:%SZ')
            params['end_time'] = end_time

        # Split `max_results` into batches of at most 500, as this is the
        # maximum number allowed
        batch_results = []
        while max_results > 500:
            batch_results.append(500)
            max_results -= 500
        batch_results.append(max_results)
        logger.debug(f'Processing batches {batch_results}')

        # Initialise dataframes
        tweet_df = pd.DataFrame()
        user_df = pd.DataFrame()
        media_df = pd.DataFrame()
        reply_df = pd.DataFrame()
        poll_df = pd.DataFrame()
        place_df = pd.DataFrame()
        meta_df = pd.DataFrame()

        # Initialise progress bar
        if len(batch_results) > 1 and progress_bar:
            pbar = tqdm(total=sum(batch_results), desc='Fetching tweets')

        for batch_result in batch_results:

            # Add the `max_results` parameter to the GET request
            params['max_results'] = batch_result

            # Perform the GET request
            response = requests.get(self.base_url,
                                    params=params,
                                    headers=self.headers)

            # If we have reached the API limit then wait a bit and try again
            while response.status_code in [429, 503]:
                logger.debug('Request limit reached. Waiting...')
                time.sleep(1)
                response = requests.get(self.base_url,
                                        params=params,
                                        headers=self.headers)

            if len(batch_results) > 1 and progress_bar:
                pbar.set_description('Fetching tweets')

            # If the GET request failed, then stop and output the status code
            if response.status_code != 200:
                raise RuntimeError(f'[{response.status_code}] {response.text}')

            # Convert the response to a dict
            data_dict = response.json()

            # If the query returned errors, then raise an exception
            if 'data' not in data_dict and 'errors' in data_dict:
                error = data_dict['errors'][0]
                raise RuntimeError(error["detail"])

            # Tweet dataframe
            if 'data' in data_dict:
                df = pd.json_normalize(data_dict['data'])
                df.set_index('id', inplace=True)
                tweet_df = pd.concat((tweet_df, df))

            # User dataframe
            if 'includes' in data_dict and 'users' in data_dict['includes']:
                users = data_dict['includes']['users']
                df = pd.json_normalize(users)
                df.set_index('id', inplace=True)
                user_df = pd.concat((user_df, df))

            # Media dataframe
            if 'includes' in data_dict and 'media' in data_dict['includes']:
                media = data_dict['includes']['media']
                df = pd.json_normalize(media)
                df.set_index('media_key', inplace=True)
                media_df = pd.concat((media_df, df))

            # Reply dataframe
            if 'includes' in data_dict and 'tweets' in data_dict['includes']:
                replies = data_dict['includes']['tweets']
                df = pd.json_normalize(replies)
                df.set_index('id', inplace=True)
                reply_df = pd.concat((reply_df, df))

            # Poll dataframe
            if 'includes' in data_dict and 'polls' in data_dict['includes']:
                polls = data_dict['includes']['polls']
                df = pd.json_normalize(polls)
                df.set_index('id', inplace=True)
                poll_df = pd.concat((poll_df, df))

            # Places dataframe
            if 'includes' in data_dict and 'places' in data_dict['includes']:
                places = data_dict['includes']['places']
                df = pd.json_normalize(places)
                df.set_index('id', inplace=True)
                place_df = pd.concat((place_df, df))

            # Meta dataframe
            meta_dict = data_dict['meta']
            if meta_dict['result_count'] > 0:
                meta_dict['date'] = dt.datetime.strftime(dt.datetime.utcnow(),
                                                         '%Y-%m-%dT%H:%M:%S')
                df = pd.DataFrame.from_records([meta_dict], index='date')
                meta_df = pd.concat((meta_df, df))

                # Set new `to_date`, so that we continue getting the next
                # tweets in the next batch
                to_date = pd.to_datetime(tweet_df.created_at).min()

                # If the new `to_date` is equal to `from_date` then there are
                # no more results and we stop
                if from_date == to_date:
                    break

            # If there are no more tweets then stop
            else:
                break

            # Update progress bar
            if len(batch_results) > 1 and progress_bar:
                pbar.update(batch_result)

            # Wait a second, to give the API a break
            time.sleep(1)

        # Close progress bar
        if len(batch_results) > 1 and progress_bar:
            pbar.close()

        all_dfs = dict(tweets=tweet_df, authors=user_df, media=media_df,
                       replies=reply_df, polls=poll_df, places=place_df,
                       metadata=meta_df)

        return all_dfs

    def get_followers(self, user_id: Union[int, List[int]]):

        # Make sure `user_id` is a list
        if isinstance(user_id, str) or isinstance(user_id, int):
            user_ids = [user_id]
        else:
            user_ids = user_id

        # Initialise dataframes
        user_df = pd.DataFrame()
        meta_df = pd.DataFrame()

        # Initialise parameters used in the GET requests
        params = {'user.fields': ','.join(self.user_fields),
                  'max_results': 100}

        for user_id in user_ids:

            # Define GET request endpoint
            url = (f'https://api.twitter.com/2/users/{user_id}'
                   f'/followers')

            # Perform the GET request
            response = requests.get(url,
                                    params=params,
                                    headers=self.headers)

            # If we have reached the API limit then wait a bit and try again
            while response.status_code in [429, 503]:
                logger.debug('Request limit reached. Waiting...')
                time.sleep(1)
                response = requests.get(url,
                                        params=params,
                                        headers=self.headers)

            # If the GET request failed, then stop and output the status code
            if response.status_code != 200:
                raise RuntimeError(f'[{response.status_code}] {response.text}')

            # Convert the response to a dict
            data_dict = response.json()

            # If the query returned errors, then raise an exception
            if 'data' not in data_dict and 'errors' in data_dict:
                error = data_dict['errors'][0]
                if (('User has been suspended' in error['detail']) or
                    ('Could not find' in error['detail']) or
                    ('not authorized to see' in error['detail'])):
                    continue
                else:
                    raise RuntimeError(error["detail"])

            # User dataframe
            if 'data' in data_dict:
                users = data_dict['data']
                df = pd.json_normalize(users)
                df.set_index('id', inplace=True)

                # Add `followee` column
                df['followee'] = [user_id for _ in range(len(df))]

                user_df = pd.concat((user_df, df))

            # Meta dataframe
            meta_dict = data_dict['meta']
            if meta_dict['result_count'] > 0:
                meta_dict['date'] = dt.datetime.strftime(dt.datetime.utcnow(),
                                                         '%Y-%m-%dT%H:%M:%S')
                df = pd.DataFrame.from_records([meta_dict], index='date')
                meta_df = pd.concat((meta_df, df))

        all_dfs = dict(users=user_df, metadata=meta_df)

        return all_dfs


    def get_followees(self, user_id: Union[int, List[int]]):

        # Make sure `user_id` is a list
        if isinstance(user_id, str) or isinstance(user_id, int):
            user_ids = [user_id]
        else:
            user_ids = user_id

        # Initialise dataframes
        user_df = pd.DataFrame()
        meta_df = pd.DataFrame()

        # Initialise parameters used in the GET request
        params = {'user.fields': ','.join(self.user_fields),
                  'max_results': 100}

        for user_id in user_ids:

            # Define GET request endpoint
            url = (f'https://api.twitter.com/2/users/{user_id}'
                   f'/following')

            # Perform the GET request
            response = requests.get(url,
                                    params=params,
                                    headers=self.headers)

            # If we have reached the API limit then wait a bit and try again
            while response.status_code in [429, 503]:
                logger.debug('Request limit reached. Waiting...')
                time.sleep(1)
                response = requests.get(url,
                                        params=params,
                                        headers=self.headers)

            # If the GET request failed, then stop and output the status code
            if response.status_code != 200:
                raise RuntimeError(f'[{response.status_code}] {response.text}')

            # Convert the response to a dict
            data_dict = response.json()

            # If the query returned errors, then raise an exception
            if 'data' not in data_dict and 'errors' in data_dict:
                error = data_dict['errors'][0]
                if (('User has been suspended' in error['detail']) or
                    ('Could not find' in error['detail']) or
                    ('not authorized to see' in error['detail'])):
                    continue
                else:
                    raise RuntimeError(error["detail"])

            # User dataframe
            if 'data' in data_dict:
                users = data_dict['data']
                df = pd.json_normalize(users)
                df.set_index('id', inplace=True)

                # Add `follower` column
                df['follower'] = [user_id for _ in range(len(df))]

                user_df = pd.concat((user_df, df))

            # Meta dataframe
            meta_dict = data_dict['meta']
            if meta_dict['result_count'] > 0:
                meta_dict['date'] = dt.datetime.strftime(dt.datetime.utcnow(),
                                                         '%Y-%m-%dT%H:%M:%S')
                df = pd.DataFrame.from_records([meta_dict], index='date')
                meta_df = pd.concat((meta_df, df))

        all_dfs = dict(users=user_df, metadata=meta_df)

        return all_dfs

    def get_retweets(self, tweet_id: Union[int, List[int]]):

        # Make sure `tweet_id` is a list
        if isinstance(tweet_id, str) or isinstance(tweet_id, int):
            tweet_ids = [tweet_id]
        else:
            tweet_ids = tweet_id

        # Initialise dataframes
        user_df = pd.DataFrame()
        meta_df = pd.DataFrame()

        # Initialise parameters used in the GET requests
        params = {'user.fields': ','.join(self.user_fields)}

        for tweet_id in tweet_ids:

            # Define GET request endpoint
            url = f'https://api.twitter.com/2/tweets/{tweet_id}/retweeted_by'

            # Perform the GET request
            while True:
                try:
                    response = requests.get(url,
                                            params=params,
                                            headers=self.headers)
                    break
                except TimeoutError:
                    time.sleep(1)

            # If we have reached the API limit then wait a bit and try again
            while response.status_code in [429, 503]:
                logger.debug('Request limit reached. Waiting...')
                time.sleep(1)
                response = requests.get(url,
                                        params=params,
                                        headers=self.headers)

            # If the GET request failed, then stop and output the status code
            if response.status_code != 200:
                raise RuntimeError(f'[{response.status_code}] {response.text}')

            # Convert the response to a dict
            data_dict = response.json()

            # If the query returned errors, then raise an exception
            if 'data' not in data_dict and 'errors' in data_dict:
                error = data_dict['errors'][0]
                if (('User has been suspended' in error['detail']) or
                    ('Could not find' in error['detail']) or
                    ('not authorized to see' in error['detail'])):
                    continue
                else:
                    raise RuntimeError(error["detail"])

            # User dataframe
            if 'data' in data_dict:
                users = data_dict['data']
                df = pd.json_normalize(users)
                df.set_index('id', inplace=True)

                # Add `retweeted` column
                df['retweeted'] = [tweet_id for _ in range(len(df))]

                user_df = pd.concat((user_df, df))

            # Meta dataframe
            meta_dict = data_dict['meta']
            if meta_dict['result_count'] > 0:
                meta_dict['date'] = dt.datetime.strftime(dt.datetime.utcnow(),
                                                         '%Y-%m-%dT%H:%M:%S')
                df = pd.DataFrame.from_records([meta_dict], index='date')
                meta_df = pd.concat((meta_df, df))

        all_dfs = dict(users=user_df, metadata=meta_df)

        return all_dfs

    def get_timeline(self, user_id: Union[int, List[int]]):

        # Make sure `user_id` is a list
        if isinstance(user_id, str) or isinstance(user_id, int):
            user_ids = [user_id]
        else:
            user_ids = user_id

        # Initialise dataframes
        tweet_df = pd.DataFrame()
        meta_df = pd.DataFrame()

        # Initialise parameters used in the GET requests
        params = {'tweet.fields': ','.join(self.tweet_fields)}

        for user_id in user_ids:

            # Define GET request endpoint
            url = f'https://api.twitter.com/2/users/{user_id}/tweets'

            # Perform the GET request
            response = requests.get(url, params=params, headers=self.headers)

            # If we have reached the API limit then wait a bit and try again
            while response.status_code in [429, 503]:
                logger.debug('Request limit reached. Waiting...')
                time.sleep(1)
                response = requests.get(url,
                                        params=params,
                                        headers=self.headers)

            # If the GET request failed, then stop and output the status code
            if response.status_code != 200:
                raise RuntimeError(f'[{response.status_code}] {response.text}')

            # Convert the response to a dict
            data_dict = response.json()

            # If the query returned errors, then raise an exception
            if 'data' not in data_dict and 'errors' in data_dict:
                error = data_dict['errors'][0]
                if (('User has been suspended' in error['detail']) or
                    ('Could not find' in error['detail']) or
                    ('not authorized to see' in error['detail'])):
                    continue
                else:
                    raise RuntimeError(error["detail"])

            # User dataframe
            if 'data' in data_dict:
                tweets = data_dict['data']
                df = pd.json_normalize(tweets)
                df.set_index('id', inplace=True)

                # Add `author` column
                df['author'] = [user_id for _ in range(len(df))]

                tweet_df = pd.concat((tweet_df, df))

            # Meta dataframe
            meta_dict = data_dict['meta']
            if meta_dict['result_count'] > 0:
                meta_dict['date'] = dt.datetime.strftime(dt.datetime.utcnow(),
                                                         '%Y-%m-%dT%H:%M:%S')
                df = pd.DataFrame.from_records([meta_dict], index='date')
                meta_df = pd.concat((meta_df, df))

        all_dfs = dict(tweets=tweet_df, metadata=meta_df)

        return all_dfs

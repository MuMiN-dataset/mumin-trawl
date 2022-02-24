'''Graph database wrapper'''

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError
from dotenv import load_dotenv; load_dotenv()
import os
import pandas as pd
import time
import logging
from timeout_decorator import timeout, TimeoutError


logger = logging.getLogger(__name__)


#@timeout(60 * 60)
def run_query_with_timeout(sess, query, **kwargs):
    with sess.begin_transaction() as tx:
        result = tx.run(query, **kwargs)
        data = result.data()
        tx.commit()
        return pd.DataFrame(data)


class Graph:
    '''Graph database wrapper'''
    def __init__(self, uri: str = 'bolt://localhost:17687'):
        self.uri = uri
        self.user = os.getenv('NEO4J_USER')
        self.password = os.getenv('NEO4J_PASSWORD')
        self.driver = GraphDatabase.driver(uri=uri,
                                           auth=(self.user, self.password))

    def __repr__(self) -> str:
        return f'Graph(uri={self.uri}, user={self.user})'

    def query(self, query: str, **kwargs) -> pd.DataFrame:
        '''Query the graph database.

        Args:
            query (str):
                The Cypher query.
            **kwargs:
                Keyword arguments to be used in self.driver.session().run.

        Returns:
            Pandas DataFrame: The results from the database.
        '''
        restart_counter = 0
        with self.driver.session() as sess:
            while True:
                try:
                    return run_query_with_timeout(sess, query, **kwargs)

                except (TimeoutError, StopIteration):
                    print('Timed out. Rebooting graph database...')
                    os.system('docker container restart neo4j')
                    time.sleep(5 * 60)

                except (ServiceUnavailable, OSError, SessionExpired,
                        TransientError) as e:
                    raise e
                    restart_counter += 1
                    time.sleep(1)
                    #if restart_counter >= 60:
                    #    print('Timed out. Rebooting graph database...')
                    #    os.system('docker container restart neo4j')
                    #    time.sleep(5 * 60)
                    #    restart_counter = 0

'''Graph database wrapper'''

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from dotenv import load_dotenv; load_dotenv()
import os
import pandas as pd
import time
import logging


logger = logging.getLogger(__name__)


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
        with self.driver.session() as sess:
            while True:
                try:
                    result = sess.run(query, **kwargs)
                    data = result.data()
                    return pd.DataFrame(data)
                except (ServiceUnavailable, OSError):
                    logger.info('Graph data unavailable. Trying again.')
                    time.sleep(1)

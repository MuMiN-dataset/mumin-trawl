'''Adds the keyword claim clusters to the Claim nodes in the graph database'''

from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm

from .graph import Graph


def add_claim_clusters():

    # Initialise graph connection
    graph = Graph()

    # Initialise claim clusters path
    claim_clusters_path = Path('data/claim-clusters.csv')

    # Load in the claim clusters
    claim_clusters_df = pd.read_csv(claim_clusters_path)

    # Build Cypher query
    query = '''
        MATCH (c:Claim {claim_en: $claim_en})
        SET c.keywords = $keywords,
            c.cluster_keywords = $cluster_keywords,
            c.cluster = $cluster
    '''

    for _, row in tqdm(list(claim_clusters_df.iterrows()),
                       desc='Adding claim clusters'):
        claim_en = row.claim
        cluster_keywords = row.label
        keywords = row.label_single
        cluster = int(row.cluster)
        if cluster == -1:
            cluster_keywords = ''

        graph.query(query,
                    claim_en=claim_en,
                    cluster_keywords=cluster_keywords,
                    keywords=keywords,
                    cluster=cluster)

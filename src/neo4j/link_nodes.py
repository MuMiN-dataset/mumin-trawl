'''Create similarity relations between claims, tweets and articles'''

from tqdm.auto import tqdm
import torch
from typing import List
from .graph import Graph


def compute_similarity(claims: List[str], node_type: str):
    graph = Graph()
    claim_query = '''
        UNWIND $claims as claim
        MATCH (c:Claim {claim:claim})
        RETURN c.embedding as claimEmbedding
    '''
    if node_type == 'tweet':
        node_query = '''
            UNWIND $claims as claim
            MATCH (c:Claim {claim:claim})
            WITH c
            MATCH (t:Tweet)
            WHERE datetime(t.createdAt) >= datetime(c.date) - duration({days:3}) AND
                  datetime(t.createdAt) <= datetime(c.date) AND
                  NOT exists((t)-[:SIMILAR]->(c))
            WITH t
            RETURN id(t) as node_id
            LIMIT 100
        '''
        node_emb_query = '''
            UNWIND $node_ids as nodeId
            MATCH (t:Tweet)
            WHERE id(t) = nodeId
            RETURN t.embedding as tweetEmbedding
        '''
        link_query = '''
            UNWIND $data AS data
            MATCH (c:Claim {claim:$claim})
            WITH c, data
            MATCH (t:Tweet)
            WHERE id(t) = data.node_id
            MERGE (t)-[:SIMILAR {score:data.score}]->(c)
        '''
    elif node_type == 'article':
        node_query = '''
            UNWIND $claims AS claim
            MATCH (c:Claim {claim:claim})
            WITH c
            MATCH (a:Article)
            WHERE NOT exists(a.notParseable) AND
                  exists(a.embedding) AND
                  datetime(a.publish_date) >= datetime(c.date) - duration({days:3}) AND
                  datetime(a.publish_date) <= datetime(c.date) AND
                  NOT exists((a)-[:SIMILAR]->(c))
            WITH a
            RETURN id(a) AS node_id
            LIMIT 100
        '''
        node_emb_query = '''
            UNWIND $node_ids as nodeId
            MATCH (a:Article)
            WHERE id(a) = nodeId
            RETURN a.embedding as nodeEmbedding
        '''
        link_query = '''
            UNWIND $data AS data
            MATCH (c:Claim {claim:$claim})
            WITH c, data
            MATCH (a:Article)
            WHERE id(a) = data.node_id
            MERGE (a)-[:SIMILAR {score:data.score}]->(c)
        '''
    else:
        raise RuntimeError('Node type not recognised!')

    result = graph.query(node_query, claims=claims)
    if len(result):
        node_ids = result.node_id.tolist()
        node_ids = torch.LongTensor(node_ids).cuda()
        node_embeddings = (graph.query(node_emb_query,
                                       node_ids=node_ids.tolist())
                                 .nodeEmbedding
                                 .tolist())
        node_embeddings = torch.FloatTensor(node_embeddings).cuda()

        claim_embedding = (graph.query(claim_query, claims=claims)
                                .claimEmbedding
                                .tolist())
        claim_embedding = torch.FloatTensor(claim_embedding).cuda()

        dot = claim_embedding @ node_embeddings.t()
        claim_norm = torch.norm(claim_embedding, dim=1).unsqueeze(1)
        node_norm = torch.norm(node_embeddings, dim=1).unsqueeze(0)
        similarities = torch.div(torch.div(dot, claim_norm), node_norm)

        for claim_idx in range(similarities.size(0)):
            similar_nodes = torch.nonzero(similarities[claim_idx] > 0.7)
            if similar_nodes.size(0) > 0:
                new_node_ids = node_ids[similar_nodes].squeeze(1).tolist()
                new_scores = (similarities[claim_idx][similar_nodes]
                              .squeeze(1)
                              .tolist())
                data = [dict(node_id=int(node_id), score=float(score))
                        for node_id, score in zip(new_node_ids, new_scores)]
                graph.query(link_query, data=data, claim=claims[claim_idx])


def link_nodes(start_from_scratch: bool = False,
               batch_size: int = 100,
               node_type: str = 'tweet'):
    # Initialise graph
    graph = Graph()

    # Remove all SIMILAR relations if `start_from_scratch` is True
    if start_from_scratch:
        graph.query('CALL apoc.periodic.iterate('
                    '"MATCH ()-[r:SIMILAR]-() RETURN r",'
                    '"DELETE r",'
                    '{batchsize:10000, parallel:false})')

    # Create query that generates all the node pairs that need to be evaluated
    if node_type == 'tweet':
        claim_query = '''
            MATCH (c:Claim)
            WHERE NOT exists(c.parsed)
            WITH c
            OPTIONAL MATCH (c)<-[r:SIMILAR]-(t:Tweet)
            WHERE r.score >= 0.7
            WITH c, count(t) as numLinks
            WHERE numLinks < 1
            SET c.parsed = true
            RETURN c.claim AS claim
            LIMIT 1
        '''
    elif node_type == 'article':
        claim_query = '''
            MATCH (c:Claim)
            WHERE NOT exists(c.parsed)
            WITH c
            OPTIONAL MATCH (c)<-[r:SIMILAR]-(a:Article)
            WHERE r.score >= 0.7
            WITH c, count(a) AS numLinks
            WHERE numLinks < 1
            SET c.parsed = true
            RETURN c.claim as claim
            LIMIT 1
        '''
    else:
        raise RuntimeError('Node type not recognised!')

    # Initialise count queries
    total_count_query = '''
        MATCH (c:Claim)
        RETURN count(c.claim) AS num
    '''
    parsed_count_query = '''
        MATCH (c:Claim)
        WHERE exists(c.parsed)
        RETURN count(c.claim) AS num
    '''
    reset_claims_query = '''
        UNWIND $claims AS claim
        MATCH (c:Claim {claim:claim})
        WHERE exists(c.parsed)
        REMOVE c.parsed
    '''

    # Initialise progress bar
    pbar = tqdm(total=graph.query(total_count_query).num[0],
                desc='Linking claims')
    num_parsed = graph.query(parsed_count_query).num[0]
    pbar.update(num_parsed)

    while True:
        claim_df = graph.query(claim_query)
        if len(claim_df) == 0:
            break

        claims = claim_df.claim.tolist()

        try:
            compute_similarity(claims, node_type=node_type)
        except:
            graph.query(reset_claims_query, claims=claims)

        # Update the progress bar
        old_parsed = int(num_parsed)
        num_parsed = graph.query(parsed_count_query).num[0]
        pbar.update(num_parsed - old_parsed)
        pbar.total = graph.query(total_count_query).num[0]
        pbar.refresh()


def add_label_nodes(start_from_scratch: bool = False):

    # Initialise a graph database connection
    graph = Graph()

    if start_from_scratch:
        graph.query('CALL apoc.periodic.iterate('
                    '"MATCH (label:Label) RETURN label",'
                    '"DETACH DELETE label",'
                    '{batchsize:1000, parallel:false})')

    # Define a dictionary with the three dataset sizes
    sizes = dict(small=0.80, medium=0.75, large=0.70)

    # Loop over the three dataset sizes
    for name, threshold in tqdm(sizes.items(), desc='Creating labels'):

        # This query fetches a single claim which has been linked but not yet
        # labelled
        get_unlabelled_claim_query = '''
            MATCH (c:Claim)<-[r:SIMILAR]-()
            WHERE r.score > $threshold AND
                  NOT exists((c)-[:HAS_LABEL]->(:Label {size:$name}))
            WITH c
            MATCH (c)<-[r:HAS_REVIEWED]-(:Reviewer)
            WHERE r.predicted_verdict IN ['misinformation', 'factual']
            WITH DISTINCT c,
                 count(DISTINCT r.predicted_verdict) AS numVerdicts
            WHERE numVerdicts = 1
            RETURN DISTINCT id(c) AS claim_id
            LIMIT 1
        '''

        # This query assigns a label to a given claim
        label_claim_query = '''
            MATCH (claim:Claim)<-[r:HAS_REVIEWED]-(:Reviewer)
            WHERE id(claim) = $claim_id AND
                  r.predicted_verdict IN ['misinformation', 'factual'] AND
                  NOT exists((claim)-[:HAS_LABEL]->(:Label {size:$name}))
            WITH DISTINCT claim,
                 count(DISTINCT r.predicted_verdict) AS numVerdicts
            WHERE numVerdicts = 1
            MATCH (claim)<-[r:HAS_REVIEWED]-(:Reviewer)
            WHERE r.predicted_verdict IN ['misinformation', 'factual']
            WITH DISTINCT claim,
                 r.predicted_verdict AS verdict
            LIMIT 1
            MERGE (claim)-[:HAS_LABEL]->(:Label {size:$name, verdict:verdict})
        '''

        # This query propagates the labels for a claim to other claims which
        # are connected to the same tweets
        propagate_labels_query = '''
            MATCH (label:Label {size:$name})<-[:HAS_LABEL]-(claim:Claim)
            WHERE id(claim) = $claim_id
            WITH label, claim
            MATCH (claim)<-[r:SIMILAR|SIMILAR_VIA_ARTICLE]-(tweet:Tweet)
            WHERE r.score > $threshold
            WITH label, claim, tweet
            MATCH (claim2:Claim)<-[r:SIMILAR|SIMILAR_VIA_ARTICLE]-(tweet)
            WHERE r.score > $threshold AND
                  NOT exists((claim2)-[:HAS_LABEL]->(:Label {size:$name}))
            MERGE (claim2)-[:HAS_LABEL]->(label)
        '''

        while True:
            # Try to fetch an unlabelled claim
            claim_id_df = graph.query(get_unlabelled_claim_query,
                                      threshold=threshold,
                                      name=name)

            # If there are no more unlabelled claims then stop
            if len(claim_id_df) == 0:
                break

            # Otherwise process the unlabelled claim
            else:
                claim_id = claim_id_df.claim_id.tolist()[0]

                # Create a label for the claim
                graph.query(label_claim_query, claim_id=claim_id, name=name)

                # Propagate the same label for connected claims
                graph.query(propagate_labels_query,
                            claim_id=claim_id,
                            name=name,
                            threshold=threshold)


def train_val_test_split(start_from_scratch: bool = False):

    # Initialise a graph database connection
    graph = Graph()

    # Remove all mask attributes if `start_from_scratch` is True
    if start_from_scratch:
        graph.query('MATCH (l:Label) '
                    'REMOVE l.train_mask, l.val_mask, l.test_mask')

    # Define a dictionary with the three dataset sizes
    sizes = dict(small=0.80, medium=0.75, large=0.70)

    # Loop over the three dataset sizes
    for name, threshold in tqdm(sizes.items(), desc='Splitting labels'):

        # Count the number of labels in the dataset
        count_labels_query = '''
            MATCH (label:Label {size:$name})
            RETURN count(DISTINCT label) AS num_labels
        '''
        num_labels = (graph.query(count_labels_query, name=name)
                           .num_labels
                           .tolist()[0])

        # Set the minimum number of labels in the validation and test set
        min_val = int(num_labels * 0.1)
        min_test = int(num_labels * 0.1)

        # Get a dataframe of the clusters in the dataset, ordered by the number
        # of labels they are connected to
        cluster_query = '''
            MATCH (c:Claim)-[:HAS_LABEL]->(label:Label {size:$name})
            WHERE exists(c.cluster)
            RETURN DISTINCT c.cluster AS cluster,
                   count(DISTINCT label) AS num_labels
            ORDER BY num_labels
        '''
        clusters = (graph.query(cluster_query, name=name)
                         .astype(dict(cluster=int, num_labels=int)))

        # Define assignment query
        assign_split_query = '''
            MATCH (c:Claim {cluster:$cluster})-[:HAS_LABEL]->(label:Label {size:$name})
            WHERE NOT exists(label.train_mask)
            SET label.train_mask = $train,
                label.val_mask = $val,
                label.test_mask = $test
        '''

        # Assign labels to the three splits, ensuring they come from different
        # clusters
        num_val = 0
        num_test = 0
        for _, row in clusters.iterrows():

            # Always assign the "no cluster" cluster to the training set
            if row.cluster == -1:
                graph.query(assign_split_query,
                            cluster=int(row.cluster),
                            name=name,
                            train=True,
                            val=False,
                            test=False)

            # Otherwise, if we haven't reached the minimum number of validation
            # samples yet, then assign the cluster to the validation set
            elif num_val < min_val:
                graph.query(assign_split_query,
                            cluster=int(row.cluster),
                            name=name,
                            train=False,
                            val=True,
                            test=False)
                num_val += (clusters.query('cluster == @row.cluster')
                                    .num_labels
                                    .tolist()[0])

            # Otherwise, if we haven't reached the minimum number of test
            # samples yet, then assign the cluster to the test set
            elif num_test < min_test:
                graph.query(assign_split_query,
                            cluster=int(row.cluster),
                            name=name,
                            train=False,
                            val=False,
                            test=True)
                num_test += (clusters.query('cluster == @row.cluster')
                                     .num_labels
                                     .tolist()[0])

            # Otherwise assign the cluster to the training dataset
            else:
                graph.query(assign_split_query,
                            cluster=int(row.cluster),
                            name=name,
                            train=True,
                            val=False,
                            test=False)

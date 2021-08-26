'''Create similarity relations between claims, tweets and articles'''

from tqdm.auto import tqdm
import torch
from .graph import Graph


def compute_similarity(claims, node_type: str):
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
            LIMIT 10000
            RETURN id(t) as tweetId
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
            UNWIND $claims as claim
            MATCH (c:Claim {claim:claim})
            WITH c
            MATCH (a:Article)
            WHERE NOT exists(a.notParseable) AND
                  exists(a.embedding) AND
                  datetime(a.publish_date) >= datetime(c.date) - duration({days:3}) AND
                  datetime(a.publish_date) <= datetime(c.date) AND
                  NOT exists((a)-[:SIMILAR]->(c))
            WITH a
            LIMIT 10000
            RETURN id(a) as nodeId
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

    node_ids = graph.query(node_query, claims=claims).nodeId.tolist()
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
        similar_nodes = torch.nonzero(similarities[claim_idx] > 0.5)
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
            WITH c
            OPTIONAL MATCH (c)<-[r:SIMILAR]-(t:Tweet)
            WHERE r.score >= 0.7
            WITH c, count(t) as numLinks
            WHERE numLinks < 1
            RETURN c.claim as claim
        '''
    elif node_type == 'article':
        claim_query = '''
            MATCH (c:Claim)
            WITH c
            OPTIONAL MATCH (c)<-[r:SIMILAR]-(a:Article)
            WHERE r.score >= 0.7
            WITH c, count(a) as numLinks
            WHERE numLinks < 1
            RETURN c.claim as claim
        '''
    else:
        raise RuntimeError('Node type not recognised!')

    while True:
        claims = graph.query(claim_query).claim.tolist()
        if len(claims) == 0: break
        batches = range(0, len(claims), batch_size)
        for idx in tqdm(batches, desc='Linking claims with nodes'):
            compute_similarity(claims[idx:idx+batch_size], node_type=node_type)

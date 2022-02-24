'''Summarise the content of all the articles in the graph database'''

from transformers import BartTokenizer, BartForConditionalGeneration
from tqdm.auto import tqdm
import warnings
from .graph import Graph


def summarise_articles(start_from_scratch: bool = False):

    # Initialise the graph database
    graph = Graph()

    # Remove all summaries from the graph
    if start_from_scratch:
        query = '''
            MATCH (a:Article)
            WHERE exists(a.summary)
            REMOVE a.summary
        '''
        graph.query(query)

    # Load the summarisation model and its tokeniser
    transformer = 'facebook/bart-large-cnn'
    tokeniser = BartTokenizer.from_pretrained(transformer)
    model = BartForConditionalGeneration.from_pretrained(transformer).cuda()

    # Define the cypher query used to get the next article
    get_article_query = '''
        MATCH (a:Article)
        WHERE exists(a.title_en) AND
              exists(a.content_en) AND
              NOT exists(a.summary) AND
              NOT exists(a.beingSummarised)
        SET a.beingSummarised = true
        RETURN a.url as url, a.title_en as title, a.content_en as content
        LIMIT 100
    '''

    # Define the cypher query used to set the summary on the Article node
    set_summary_query = '''
        UNWIND $url_summaries as url_summary
        MATCH (a:Article {url:url_summary.url})
        REMOVE a.beingSummarised
        SET a.summary = url_summary.summary
    '''

    # Define the cypher query used to count the remaining articles
    total_count_query = '''
        MATCH (a:Article)
        RETURN count(a) as num_articles
    '''
    summarised_count_query = '''
        MATCH (a:Article)
        WHERE exists(a.summary)
        RETURN count(a) as num_articles
    '''
    not_summarised_count_query = '''
        MATCH (a:Article)
        WHERE not exists(a.summary)
        RETURN count(a) as num_articles
    '''

    # Get the total number of articles and define a progress bar
    num_urls = graph.query(total_count_query).num_articles[0]
    num_summarised = graph.query(summarised_count_query).num_articles[0]
    pbar = tqdm(total=num_urls, desc='Summarising articles')
    pbar.update(num_summarised)

    # Continue summarising until all articles have been summnarised
    while graph.query(not_summarised_count_query).num_articles[0]> 0:

        # Fetch new articles
        article_df = graph.query(get_article_query)
        urls = article_df.url.tolist()
        docs = [row.title + '\n' + row.content
                for _, row in article_df.iterrows()]

        # Tokenise the content of the articles
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            # Extract the summary of the articles
            summaries = list()
            for doc in docs:
                tokens = tokeniser([doc], return_tensors='pt', padding=True,
                                   truncation=True, max_length=1000)

                summary_ids = model.generate(tokens['input_ids'].cuda(),
                                             num_beams=4,
                                             max_length=512,
                                             early_stopping=True)

                summary = tokeniser.batch_decode(
                    summary_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]

                summaries.append(summary)

        # Set the summary as an attribute on the Article nodes
        url_summaries = [dict(url=url, summary=summary)
                         for url, summary in zip(urls, summaries)]
        graph.query(set_summary_query, url_summaries=url_summaries)

        # Update the progress bar
        old_summarised = int(num_summarised)
        num_summarised = graph.query(summarised_count_query).num_articles[0]
        pbar.update(num_summarised - old_summarised)
        pbar.total = graph.query(total_count_query).num_articles[0]
        pbar.refresh()

    # Close the progress bar
    pbar.close()

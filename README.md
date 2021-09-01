# MuMiN-Trawl
This repository contains the source code used to construct, from scratch, the
MuMiN dataset from the paper [Nielsen and McConville: _MuMiN: A Large-Scale
Multilingual Multimodal Fact-Checked Misinformation Dataset with Linked Social
Network Posts_ (2021)](https://openreview.net/forum?id=sOLdMFkQe7).

The idea behind releasing this repository is to give developers and researchers
a way to extend the MuMiN framework, or to use parts of this project to further
research in related projects.


## Code Overview
This section will give an overview of the various scripts in this repository,
to make it as easy as possible to locate the functions and classes that could
be useful for other projects.

### API Keys
To use the scripts, several API keys needs to be set up and stored in a file
called `.env`, in the root of the repository. The file should look like this:
```shell
GOOGLE_API_KEY=XXXXX
DEEPL_API_KEY=XXXXX
TWITTER_API_KEY=XXXXX
NEO4J_USER=XXXXX
NEO4J_PASSWORD=XXXXX
```

The API keys are only required when a function is used that demands it
(otherwise an error will occur).

The Google API key
[can be set up here](https://cloud.google.com/docs/authentication/api-keys),
and is used for fetching the claims and verdicts, as well as for the
translation. Thus, both the
[Fact Check Tools API](https://developers.google.com/fact-check/tools/api/reference/rest)
and the
[Cloud Translation API](https://cloud.google.com/translate/docs/reference/rest/)
needs to be enabled for the API key.

The Twitter Academic API key
[can be applied for here](https://developer.twitter.com/en/products/twitter-api/academic-research)
and is used to fetch tweets from Twitter. Note that it is the _Bearer Token_
that is used as the `TWITTER_API_KEY`.

The DeepL API key
[can be set up here](https://www.deepl.com/en/pro/change-plan#developer)
and is only used if you want to use DeepL for translation instead of Google.

The Neo4j user and password is set up when a Neo4j database is created.
Creating such a database is covered in the `Populating graph database` section
below, so you can wait with the `NEO4J_USER` and `NEO4J_PASSWORD` until the
database is set up.

### Main script
This is the script which is used to call all the other scripts. All the needed
function calls are already wrapped by functions in this script, so everything
should be callable using these wrapper functions. Choose the function to call
at the bottom of the script.

### Fetching claims
To fetch the claims and verdicts from the Google Fact Check API, we created a
wrapper, `FactChecker`, which lies in the `fact_checker.py` script. This
wrapper only has a single method, `__call__`, which can be used to easily query
claims and verdicts from the API. This uses the `GOOGLE_API_KEY` from the
`.env` file, and thus assumes that you have that set up at this point.

Before we are fetching claims and verdicts, we first need to construct a list
of fact-checking websites from which we want to fetch the claims from. This
repository already comes with a list of reviewers, which was the one compiled
when our paper was written, but this can be updated using the
`build_reviewer_list` function in the `build_reviewer_list.py` script. This
fetches fact-checking websites posting claims in each of the list of languages
present in the `data/bcp47.yaml` file, which can be manually updated if
required.

The fetching is then done with the `fetch_facts` function in the
`fetch_facts.py` script, which assumes that the `data/reviewers.yaml` file
has been created, using the procedure described in the above paragraph. This
takes a while, and the result is a folder `data/reviewers`, which contains a
`csv` file for each of the fact-checking websites. Each of these `csv` files
will then contain all the claims from that fact-checking website.

### Fetching tweets
The tweets are fetched using a Twitter API wrapper, `Twitter`, which is defined
in the `twitter.py` script. This primary method of this wrapper, `query`, which
can be used to query the Twitter Full-Archive Search API.

The tweets are then fetched primarily using the
`fetch_tweets_from_extracted_keywords` function in the `fetch_tweets.py`
script. This function assumes that the claims have been downloaded and are
present in the `data/reviewers` folder, as described in the previous section,
extracts keywords from them and queries Twitter for tweets in the claim period
related to the keywords. This uses the `fetch_tweets` function, which also lies
in `fetch_tweets.py` script, which fetches Twitter data more generally.

### Verdict classification
As described in [the paper](todo), the verdicts in the `data/reviewers` folder
will be plaintext. To convert these into our three classes `factual`,
`misinformation` and `other`, we train a separate classifier. The source code
for this model can be found in `verdict_classifier` folder.

The first step is to annotate verdicts into these three classes, which can be
done in a notebook using the `verdict_classifier/annotate.py` script and the
notebook `notebooks/annotate_verdict_classifier.ipynb`. We _do_ provide our
annotated verdicts as well, lying in the `data/verdict_annotations.csv` file.

The dataset is wrapped into a `PyTorch` dataset in the
`verdict_classifier/dataset.py` script, and the model itself is defined in the
`verdict_classifier/model.py` script. The training of the model is done using
the `train` function in the `verdict_classifier/train.py` script, which assumes
that the annotated verdicts lie in the `data/verdict_annotations.csv` file. The
resulting model will then lie in the `models` folder. Alternatively, our
trained model already lies in this folder, and can simply be used as-is.

With a trained verdict classifier, the next step is then to run the
`add_predicted_verdicts` function in the `add_predicted_verdicts.py` script,
which will add a `predicted_verdict` column to the claim `csv` files in the
`data/reviewers` folder.

### Populating graph database
Next, we need to populate a Neo4j graph database with all the data fetched so
far. The easiest way to construct such a database is using
[Docker](https://docs.docker.com/get-docker/). Once Docker is installed, the
graph database can be launched using the `neo4j/launch.sh` shell script. After
a few minutes, you can access the database through your browser at
<https://localhost:17474>. In the top field, `Connect URL`, select `neo4j://`
and insert `localhost:17687`. The default username is `neo4j` and the default
password is also `neo4j`. When logging in for the first time, a different
password will need to be set. After having set a new password, insert the
username and password as the `NEO4J_USER` and `NEO4J_PASSWORD` in your `.env`
files, as these will be used below.

The querying of the graph database is done using a Neo4j wrapper, `Graph`, in
the `neo4j/graph.py` script. As the previous wrappers it has a single method,
`query`, which queries the database with a
[Cypher script](https://neo4j.com/developer/cypher/).

To populate the graph database with the downloaded Twitter data as well as the
claims, the functions `populate` and `create_claims` in the `neo4j/populate.py`
script are used. This will use all the Cypher scripts in the `neo4j/cypher`
folder, which creates all the nodes and relations, enforces uniqueness
constraints and creates search indices. Note that this will also extract the
URLs from the tweets and store these as separate `Url` nodes.

### Translating claims and tweets
As we are doing a lot of translation in building this dataset, we built a
dedicated Google Cloud Translation API wrapper, `GoogleTranslator`, which lies
in the `translator.py` script. We also tried an alternative translation
provider [DeepL](https://www.deepl.com/translator) and have a wrapper,
`DeepLTranslator`, for that in the same script.

The translation of the claims, tweets and articles are done using the
`translate_claims`, `translate_tweets` and `translate_articles` functions in
the `neo4j/translate.py` script. This assumes that you have populated these
into the graph database. This will store a `text_en` attribute on all the
`Tweet` nodes and a `claim_en` attribute on the `Claim` nodes.

### Fetching and processing articles
The fetching of the articles is done with the `create_articles` function in the
`neo4j/populate.py` script, which fetches all the URLs from the graph database,
filters them and extracts the content using the `newspaper3k` Python library.
These articles will then be stored as `Article` nodes in the database.

The next step is translating the articles, which is done using the
`translate_articles` function in the `neo4j/translate.py` script. This is done
a bit differently than the other translation functions, as text limitations in
the Google Cloud Translation API forces us to split the article content into
sentences first, translate each sentence and concatenate the resulting
translated sentences. This stores `title_en` and `content_en` on all the
`Article` nodes in the database.

Next, the articles need to be summarised, which is done using the
`summarise_articles` function in the `summarise.py` script. This takes a long
time (summarising a million articles takes several weeks), and the result is a
`summary` attribute on all the `Article` nodes.

### Embedding claims, tweets and articles
With translated claims, tweets and articles, we next need to embed these, to be
able to measure similarity between them. This is done using the `embed_claims`,
`embed_tweets` and `embed_articles` functions in the `neo4j/embed.py` script.
This results in an `embedding` attribute on all the `Claim`, `Tweet` and
`Article` nodes.

### Linking claims with tweets and articles
With embeddings on all the relevant nodes, we next link the claims with the
tweets and articles, using the `link_nodes` function in the
`neo4j/link_nodes.py` script. In this function we can specify the `node_type`
argument to either be `tweet` or `article`, depending on whether we are linking
tweets or articles with claims. This script continues until all `Claim` nodes
have been linked to a `Tweet` node (if `node_type` is `tweet`) or an `Article`
node (if `node_type` is `article`). This will probably never happen, so run the
script until you are satisfied with the amount of links. This will usually take
several weeks.

### Dumping the graph database
With the links in place, the public database (i.e., tweet IDs rather than the
tweets themselves) can then be dumped using the `dump_database` function in the
`neo4j/dump_database.py` script. The folder containing the dump needs to be
specified by changing the `dump_dir` variable in this script.

### Fetching replies and social network
In our database we only dealt with the _source tweets_, being the tweets that
initiates a Twitter thread, and did not include any of the retweets, as this
would simply require too much memory. Instead, now that we have the tweet IDs
of the source tweets and user IDs of the final dataset, we can add in this
information using the `fetch_retweets`, `fetch_followers` and `fetch_followees`
functions in the `neo4j/finalise_tweet_data.py` script. Again, the `dump_dir`
variable in this script needs to be changed to where the database dump is
located.


## Related Repositories
- [MuMiN](https://github.com/CLARITI-REPHRAIN/mumin), containing the
  paper in PDF and LaTeX form.
- [MuMiN-Build](https://github.com/CLARITI-REPHRAIN/mumin-build),
  containing the scripts for the Python package `mumin`, used to compile the
  dataset and export it to various graph machine learning frameworks.

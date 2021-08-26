UNWIND $tweet_urls AS tweet
UNWIND tweet.urls AS url
WITH tweet, url
WHERE tweet.idx IS NOT NULL AND url IS NOT NULL
MATCH (t:Tweet {tweetId:tweet.idx})
MATCH (u:Url {name:url})
MERGE (t)-[:LINKS_TO]->(u);

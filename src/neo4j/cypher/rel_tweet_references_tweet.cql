UNWIND $tweets AS tweet
WITH tweet
WHERE tweet.id IS NOT NULL AND tweet.referenced_tweets IS NOT NULL
UNWIND apoc.convert.fromJsonList(tweet.referenced_tweets) as refTweetId
WITH tweet.id as tweetId, refTweetId
WHERE refTweetId IS NOT NULL
MATCH (t1:Tweet {tweetId:tweetId})
MATCH (t2:Tweet {tweetId:refTweetId})
MERGE (t1)-[:REFERENCES]->(t2);

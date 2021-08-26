UNWIND $tweets AS tweet
WITH tweet
WHERE tweet['entities.hashtags'] IS NOT NULL AND tweet.id IS NOT NULL
UNWIND apoc.convert.fromJsonList(tweet['entities.hashtags']) as tag
WITH tweet.id as tweetId, toLower(tag.tag) as tag
WHERE tag IS NOT NULL
MATCH (t:Tweet {tweetId:tweetId})
MATCH (h:Hashtag {name:tag})
MERGE (t)-[:HAS_HASHTAG]->(h);

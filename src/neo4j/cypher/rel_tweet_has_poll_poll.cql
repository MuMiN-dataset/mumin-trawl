UNWIND $tweets AS tweet
WITH tweet
WHERE tweet.id IS NOT NULL AND tweet['attachments.poll_ids'] IS NOT NULL
UNWIND apoc.convert.fromJsonList(tweet['attachments.poll_ids']) as pollId
WITH tweet.id AS tweetId, pollId
WHERE pollId IS NOT NULL
MATCH (t:Tweet {tweetId:tweetId})
MATCH (p:Poll {pollId:pollId})
MERGE (t)-[:HAS_POLL]->(p);

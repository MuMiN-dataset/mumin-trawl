UNWIND $tweets AS tweet
WITH tweet
WHERE tweet.id IS NOT NULL AND tweet['attachments.media_keys'] IS NOT NULL
UNWIND apoc.convert.fromJsonList(tweet['attachments.media_keys']) as mediaId
WITH tweet.id as tweetId, mediaId
WHERE mediaId IS NOT NULL
MATCH (t:Tweet {tweetId:tweetId})
MATCH (m:Media {mediaId:mediaId})
MERGE (t)-[:HAS_MEDIA]->(m);

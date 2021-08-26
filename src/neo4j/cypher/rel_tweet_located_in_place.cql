UNWIND $tweets AS tweet
WITH tweet
WHERE tweet.id IS NOT NULL AND tweet['geo.place_id'] IS NOT NULL
MATCH (t:Tweet {tweetId:tweet.id})
MATCH (p:Place {placeId:tweet['geo.place_id']})
MERGE (t)-[:LOCATED_IN]->(p);

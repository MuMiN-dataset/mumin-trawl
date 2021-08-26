UNWIND $tweets AS tweet
WITH tweet
WHERE tweet.id IS NOT NULL and tweet.author_id IS NOT NULL
MATCH (t:Tweet {tweetId:tweet.id})
MATCH (u:User {userId:tweet.author_id})
MERGE (u)-[:POSTED]->(t);

UNWIND $tweets AS tweet
WITH tweet
WHERE tweet.id IS NOT NULL AND tweet.conversation_id  IS NOT NULL
MATCH (t:Tweet {tweetId:tweet.id})
MATCH (c:Conversation {conversationId:tweet.conversation_id})
MERGE (t)-[:IS_PART_OF]->(c);

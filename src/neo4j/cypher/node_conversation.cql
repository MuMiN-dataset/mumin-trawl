UNWIND $tweets AS tweet
WITH DISTINCT tweet.conversation_id as conversationId
WHERE conversationId IS NOT NULL
MERGE (n:Conversation {conversationId:conversationId});

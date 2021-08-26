UNWIND $tweets AS tweet
WITH tweet.id as tweetId, tweet['entities.mentions'] as mentions
WHERE tweetId IS NOT NULL AND mentions IS NOT NULL
UNWIND apoc.convert.fromJsonList(mentions) as mentionedUser
WITH tweetId, mentionedUser.username as mentionedUserName
WHERE mentionedUserName IS NOT NULL
MATCH (t:Tweet {tweetId:tweetId})
MATCH (u:User {userName:toLower(mentionedUserName)})
MERGE (t)-[:MENTIONS]->(u);

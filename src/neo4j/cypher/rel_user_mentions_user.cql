UNWIND $users AS user
WITH user.id as userId, user['entities.description.mentions'] as mentions
WHERE userId IS NOT NULL AND mentions IS NOT NULL
UNWIND apoc.convert.fromJsonList(mentions) as mentionedUser
WITH userId, mentionedUser.username as mentionedUserName
WHERE mentionedUserName IS NOT NULL
MATCH (u1:User {userId:userId})
MATCH (u2:User {userName:toLower(mentionedUserName)})
MERGE (u1)-[:MENTIONS]->(u2);

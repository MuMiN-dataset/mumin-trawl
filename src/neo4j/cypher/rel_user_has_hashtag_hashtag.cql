UNWIND $users AS user
WITH user
WHERE user.id IS NOT NULL AND user['entities.description.hashtags'] IS NOT NULL
UNWIND apoc.convert.fromJsonList(user['entities.description.hashtags']) as tag
WITH user.id as userId, toLower(tag.tag) as tag
WHERE tag IS NOT NULL
MATCH (u:User {userId:userId})
MATCH (h:Hashtag {name:tag})
MERGE (u)-[:HAS_HASHTAG]->(h);

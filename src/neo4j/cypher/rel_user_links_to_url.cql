UNWIND $user_urls AS user
UNWIND user.urls as url
WITH user, url
WHERE user.idx IS NOT NULL AND url IS NOT NULL
MATCH (u:User {userId:user.idx})
MATCH (ur:Url {name:url})
MERGE (u)-[:LINKS_TO]->(ur);

UNWIND $users AS user
WITH user.id as userId, user.profile_image_url as profileImage
WHERE userId IS NOT NULL AND profileImage IS NOT NULL
MATCH (u:User {userId:userId})
MATCH (url:Url {name:profileImage})
MERGE (u)-[:HAS_PROFILE_PICTURE]->(url);

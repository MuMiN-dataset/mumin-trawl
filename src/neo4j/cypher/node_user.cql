UNWIND $users AS user
WITH user
WHERE user.username IS NOT NULL AND user.id IS NOT NULL
MERGE (n:User {userId:toInteger(user.id)})
ON CREATE SET
    n.userName = toLower(user.username),
    n.verified = toBoolean(user.verified),
    n.description = user.description,
    n.name = user.name,
    n.createdAt = user.created_at,
    n.followersCount = toInteger(user['public_metrics.followers_count']),
    n.followingCount = toInteger(user['public_metrics.following_count']),
    n.tweetCount = toInteger(user['public_metrics.tweet_count']);

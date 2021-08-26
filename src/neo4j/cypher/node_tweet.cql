UNWIND $tweets AS tweet
WITH tweet
WHERE tweet.id IS NOT NULL
MERGE (n:Tweet {tweetId:tweet.id})
ON CREATE SET
    n.text = tweet.text,
    n.source = tweet.source,
    n.lang = tweet.lang,
    n.createdAt = tweet.created_at,
    n.likeCount = toInteger(tweet['public_metrics.like_count']),
    n.retweetCount = toInteger(tweet['public_metrics.retweet_count']),
    n.replyCount = toInteger(tweet['public_metrics.reply_count']),
    n.quoteCount = toInteger(tweet['public_metrics.quote_count']);

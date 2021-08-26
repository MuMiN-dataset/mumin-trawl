UNWIND $replies AS reply
WITH reply
WHERE reply.id IS NOT NULL
MERGE (n:Tweet {tweetId:reply.id})
ON CREATE SET
    n.text = reply.text,
    n.source = reply.source,
    n.lang = reply.lang,
    n.createdAt = reply.created_at,
    n.likeCount = toInteger(reply['public_metrics.like_count']),
    n.retweetCount = toInteger(reply['public_metrics.retweet_count']),
    n.replyCount = toInteger(reply['public_metrics.reply_count']),
    n.quoteCount = toInteger(reply['public_metrics.quote_count']);

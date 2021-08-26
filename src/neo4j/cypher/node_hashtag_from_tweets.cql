UNWIND $tweets as tweet
WITH tweet['entities.hashtags'] as hashtags
WHERE hashtags IS NOT NULL
UNWIND apoc.convert.fromJsonList(hashtags) as tag
WITH DISTINCT toLower(tag.tag) AS tag
WHERE tag IS NOT NULL
MERGE (:Hashtag {name:tag});

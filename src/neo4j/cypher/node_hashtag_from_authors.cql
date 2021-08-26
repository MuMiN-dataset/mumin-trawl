UNWIND $users AS user
WITH user['entities.description.hashtags'] AS hashtags
WHERE hashtags IS NOT NULL
UNWIND apoc.convert.fromJsonList(hashtags) AS tag
WITH DISTINCT toLower(tag.tag) AS tag
WHERE tag IS NOT NULL
MERGE (:Hashtag {name:tag});

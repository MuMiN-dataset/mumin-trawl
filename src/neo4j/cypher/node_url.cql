UNWIND $urls AS urls
UNWIND urls.urls as url
WITH url
WHERE url IS NOT NULL
MERGE (:Url {name:url});

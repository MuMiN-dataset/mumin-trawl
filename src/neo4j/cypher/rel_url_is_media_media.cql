UNWIND $media AS media
WITH media
WHERE media.media_key IS NOT NULL AND
      (media.url IS NOT NULL OR media.preview_image_url IS NOT NULL)
MATCH (m:Media {mediaId:media.media_key})
MATCH (u:Url)
WHERE u.name IN [media.url, media.preview_image_url]
MERGE (u)-[:IS_MEDIA]->(m);

UNWIND $media as media
WITH media
WHERE media.media_key IS NOT NULL
MERGE (n:Media {mediaId:media.media_key})
ON CREATE SET
    n.type = media.type,
    n.height = toInteger(media.height),
    n.width = toInteger(media.width),
    n.durationMs = toInteger(media.duration_ms),
    n.viewCount = toInteger(media['public_metrics.view_count']);

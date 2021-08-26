UNWIND $places AS place
WITH place
WHERE place.id IS NOT NULL
MERGE (n:Place {placeId:toString(place.id)})
ON CREATE SET
    n.name = place.name,
    n.country_code = place.country_code,
    n.country = place.country,
    n.fullName = place.full_name,
    n.placeType = place.place_type,
    n.boundingBox = apoc.convert.fromJsonList(place['geo.bbox']),
    n.lat = apoc.coll.avg([apoc.convert.fromJsonList(place['geo.bbox'])[0],
                           apoc.convert.fromJsonList(place['geo.bbox'])[2]]),
    n.lng = apoc.coll.avg([apoc.convert.fromJsonList(place['geo.bbox'])[1],
                           apoc.convert.fromJsonList(place['geo.bbox'])[3]])

# Stop the graph database temporarily, as it is required to perform an offline
# dump
docker stop neo4j || echo "Skipping"

# Get the current date as YYYY-MM-DD, as it will be used in the dump filename
currentDate=`date +"%Y-%m-%d"`

# Dump the database via a separate backup container
docker run -it \
    --rm \
    --publish 17474:7474 \
    --publish 17687:7687 \
    --name neo4j-backup \
    --volume /media/secure/dan/neo4j/data:/data \
    --volume /media/secure/dan/neo4j_backup:/neo4j_backup \
    -e NEO4J_apoc_import_file_enabled=true \
    -e NEO4J_dbms_memory_heap_initial__size=25G \
    -e NEO4J_dbms_memory_heap_max__size=25G \
    --entrypoint="/bin/bash" \
    neo4j:4.2 \
    -c "neo4j-admin dump --to=/neo4j_backup/neo4j-$currentDate.dump"

# Restart the graph database again
docker start neo4j

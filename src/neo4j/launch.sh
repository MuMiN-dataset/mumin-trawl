# Ensure that the Neo4j folder exists
if [ ! -d /media/secure/dan/neo4j ]
then
    mkdir /media/secure/dan/neo4j
fi

# Ensure that the Neo4j data folder exists
if [ ! -d /media/secure/dan/neo4j/data ]
then
    mkdir /media/secure/dan/neo4j/data
fi

# Ensure that the Neo4j import folder exists
if [ ! -d /media/secure/dan/neo4j/import ]
then
    mkdir /media/secure/dan/neo4j/import
fi

# Ensure that the Neo4j plugins folder exists, and that APOC and GDS are in
# there
if [ ! -d /media/secure/dan/neo4j/plugins ]
then
    mkdir /media/secure/dan/neo4j/plugins
fi

# Start the Neo4j graph database as a Docker container
#    -e NEO4J_dbms_security_auth__enabled=false \
docker run -d \
    --restart unless-stopped \
    --publish 17474:7474 \
    --publish 17687:7687 \
    --name neo4j \
    --volume /media/secure/dan/twitter:/import/twitter \
    --volume /media/secure/dan/neo4j/data:/data \
    --volume /media/secure/dan/neo4j/plugins:/plugins \
    --volume $PWD/src/neo4j:/import/src \
    -e NEO4J_apoc_import_file_enabled=true \
    -e NEO4J_dbms_memory_heap_initial__size=15G \
    -e NEO4J_dbms_memory_heap_max__size=15G \
    -e NEO4JLABS_PLUGINS='["apoc", "graph-data-science"]' \
    neo4j:4.2

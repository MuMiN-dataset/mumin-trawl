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
    wget https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases/download/4.2.0.2/apoc-4.2.0.2-all.jar -P /media/secure/dan/neo4j/plugins
    wget https://s3-eu-west-1.amazonaws.com/com.neo4j.graphalgorithms.dist/graph-data-science/neo4j-graph-data-science-1.6.0-standalone.zip -P /media/secure/dan/neo4j/plugins
fi

# Start the Neo4j graph database as a Docker container
docker run -d \
    --restart unless-stopped \
    --publish 17474:7474 \
    --publish 17687:7687 \
    --volume /media/secure/dan/neo4j/data:/data \
    --volume /media/secure/dan/neo4j/plugins:/plugins \
    --volume /media/secure/dan/twitter:/import/twitter \
    --volume $PWD/src/neo4j:/import/src \
    --name neo4j \
    -e NEO4J_dbms_memory_heap_initial__size=35G \
    -e NEO4J_dbms_memory_heap_max__size=35G \
    -e NEO4J_apoc_import_file_enabled=true \
    -e NEO4JLABS_PLUGINS='["apoc", "graph-data-science"]' \
    neo4j:4.2

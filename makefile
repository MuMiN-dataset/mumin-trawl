neo4j:
	sh src/neo4j/launch.sh

jupyter:
	nohup jupyter lab \
		--autoreload \
		--port 18888 \
		--notebookdir=notebooks &

tensorboard:
	nohup tensorboard \
		--logdir tb_logs \
		--port 16006 &

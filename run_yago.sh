#!/bin/bash


cd dual_pathway
bash run_yago.sh > ../dual_pathway_yago.log 2>&1 &
dual_pathway_PID=$!
cd ..


cd HousE
bash run.sh HousE YAGO3-10 0 2 2 1800 400 1000 20 1 0.7 26.0 1.16010547465235 0.00258708538141072 250000 40000 8 0.0881968094660471
HOUSE_PID=$!
cd ../..


wait $dual_pathway_PID
wait $HOUSE_PID

echo "look log"
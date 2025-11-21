#!/bin/bash


cd dual_pathway
bash run_nell.sh > ../dual_pathway_nell.log 2>&1 &
dual_pathway_PID=$!
cd ..


cd HousE
bash run.sh HousE nell-995 1 0 0 1000 200 800 8 1 0.5 6.0 1.14940435933987 0.000575323908649059 60000 20000 8 0.0960737047401994
HOUSE_PID=$!
cd ../..


wait $dual_pathway_PID
wait $HOUSE_PID

echo "look log"
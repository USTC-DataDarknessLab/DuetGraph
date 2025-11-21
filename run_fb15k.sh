#!/bin/bash


cd dual_pathway
bash run_fb15k.sh > ../dual_pathway_fb15k.log 2>&1 &
dual_pathway_PID=$!
cd ..


cd HousE
bash run.sh HousE FB15k-237 1 2 2 500 500 600 20 6 0.6 5.0 2.00378388680359 0.000794267891285676 100000 10000 16 0.00336727231946076 
HOUSE_PID=$!
cd ../..


wait $dual_pathway_PID
wait $HOUSE_PID

echo "look log"
#!/bin/bash


cd dual_pathway
bash run_win18_v1.sh > ../dual_pathway_win18_v1.log 2>&1 &
dual_pathway_PID=$!
cd ..


cd RED-GNN/inductive
bash run_win18_v1.sh > ../../redgnn.log 2>&1 &
REDGNN_PID=$!
cd ../..


wait $dual_pathway_PID
wait $REDGNN_PID

echo "look log."
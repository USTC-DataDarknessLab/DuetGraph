#!/bin/bash


cd dual_pathway
bash run_nell_v3.sh > ../dual_pathway_nell_v3.log 2>&1 &
dual_pathway_PID=$!
cd ..


cd RED-GNN/inductive
bash run_nell_v3.sh > ../../redgnn.log 2>&1 &
REDGNN_PID=$!
cd ../..


wait $dual_pathway_PID
wait $REDGNN_PID

echo "look log."
#!/bin/bash

echo "Running matmul benchmark without loop flattening..."

python3 ./main.py

echo "Running matmul benchmark with loop flattening..."

python3 ./main.py --flatten-loops

proton-viewer -m tflop16/s -diff matmul.hatchet matmul_flatten.hatchet
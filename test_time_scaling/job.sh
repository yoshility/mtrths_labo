#!/bin/bash

indices=({1..10})

for i in "${indices[@]}"; do
    echo "Running: python main.py --index $i"
    python main.py --index $i
done
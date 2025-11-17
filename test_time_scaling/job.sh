#!/bin/bash

indices=({0..500})

for i in "${indices[@]}"; do
    echo "Running: python main.py --index $i"
    python main.py --index $i
done
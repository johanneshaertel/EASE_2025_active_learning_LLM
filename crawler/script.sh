#!/bin/bash

while true; do
    echo "New Iteration"
    python main.py || continue
done

echo "Terminated"
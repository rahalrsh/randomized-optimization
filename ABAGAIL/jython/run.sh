#!/bin/bash
# edit the classpath to to the location of your ABAGAIL jar file
#

export CLASSPATH=ABAGAIL.jar:$CLASSPATH
mkdir -p data/plot logs image

echo "continuous peaks"
jython continuouspeaks.py

echo "knapsack"
jython knapsack.py

echo "TSP"
jython travelingsalesman.py

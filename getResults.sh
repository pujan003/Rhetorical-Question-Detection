#!/bin/bash
echo -e 'RESULTS \n' > $1_evals.txt
for c in {1..20}
do
   ./final.sh $1
done
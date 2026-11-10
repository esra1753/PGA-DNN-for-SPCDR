#!/usr/bin/env bash
# Trains all models of the paper and regenerates the tables and figures.
# Each run takes about 20-25 min (PGA-DNN, PA-DNN, DD) or 1-2 h (PINN) on a CPU.
set -e
for P in TP1 TP2 TP3; do
  for S in 1234 2468 4123 5678; do
    python src/train.py --problem $P --method PGA --seed $S     # Tables 1, 3, 4, Figs. 6, 7
    python src/train.py --problem $P --method PA  --seed $S     # Table 1
  done
done
python src/train.py --problem TP1 --method PA --na 200 --seed 1234   # Table 2
python src/train.py --problem TP1 --method DD   --seed 4123          # Table 4, Fig. 7
python src/train.py --problem TP1 --method PINN --seed 4123
python src/train.py --problem TP2 --method DD   --seed 5678
python src/train.py --problem TP2 --method PINN --seed 5678
python src/train.py --problem TP3 --method DD   --seed 2468
python src/train.py --problem TP3 --method PINN --seed 2468

cd scripts
python table1.py
python table2.py
python table3.py
python table4.py
python fig_convergence.py
python fig_solutions.py

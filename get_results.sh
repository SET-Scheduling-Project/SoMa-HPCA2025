#!/bin/bash
# Usage: ./get_results.sh
python pyscripts/get_costs_overall.py results/overall/json overall.csv > stats.log
python pyscripts/get_costs_dse.py results/dse/json dse.csv
python pyscripts/Fig7_reproduce.py dse.csv Fig7_heatmaps_DSE.svg
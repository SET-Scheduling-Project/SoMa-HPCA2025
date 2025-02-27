#!/bin/bash
# Usage: ./build.sh
# Description: Build the program and create directories for results.
make clean
make release -j `nproc`

create_dir_if_not_exists() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "Directory '$dir' created."
    else
        echo "Directory '$dir' already exists."
    fi
}

rm -rf results
create_dir_if_not_exists "results"

create_dir_if_not_exists "results/dse"
create_dir_if_not_exists "results/dse/log"
create_dir_if_not_exists "results/dse/enc"
create_dir_if_not_exists "results/dse/json"

create_dir_if_not_exists "results/overall"
create_dir_if_not_exists "results/overall/log"
create_dir_if_not_exists "results/overall/enc"
create_dir_if_not_exists "results/overall/json"
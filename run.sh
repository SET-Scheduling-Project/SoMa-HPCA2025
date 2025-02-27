#!/bin/bash
# Usage: ./run.sh [--eta] [--log /path/to/log_file]
# Description: Run the program with parallel tasks and show progress with ETA (optinal).
# codes below should have the same effect as the this command:
# $> parallel --eta -a ./args.txt --colsep ' ' --noswap --joblog run.log --jobs `nproc` --linebuffer ./build/soma

total_tasks=$(wc -l < args.txt)
log_file="run.log"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --log)
            shift
            log_file="$1"
            ;;
        --eta)
            eta_enabled=true
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done

echo "" > "$log_file"

show_progress() {
    format_time() {
        local T=$1
        printf '%02d:%02d:%02d' $((T/3600)) $(( (T/60)%60 )) $((T%60))
    }

    local start_time=$(date +%s)

    while true; do
        local completed_tasks=$(grep -c "Exit:" "$log_file")
        local remaining_tasks=$((total_tasks - completed_tasks))
        local current_time=$(date +%s)
        local elapsed_time=$((current_time - start_time))

        if [ "$completed_tasks" -gt 0 ]; then
            avg_time_per_task=$(awk -v et="$elapsed_time" -v ct="$completed_tasks" 'BEGIN {printf "%.2f", et / ct}')
            total_estimated_time=$(awk -v avg="$avg_time_per_task" -v tt="$total_tasks" 'BEGIN {printf "%.0f", avg * tt}')
            eta=$((total_estimated_time - elapsed_time))
            if [ "$eta" -lt 0 ]; then
                eta=0
            fi
            eta_formatted=$(format_time "$eta")
            progress_msg="Progress: $completed_tasks/$total_tasks completed. ETA: $eta_formatted"
        else
            progress_msg="Progress: 0/$total_tasks completed. ETA: Calculating..."
        fi

        echo -ne "\r\033[K$progress_msg"

        if [ "$remaining_tasks" -le 0 ]; then
            echo -e "\nAll tasks completed!"
            break
        fi
        sleep 1
    done
}

start_parallel_tasks() {
    xargs -a args.txt -P $(nproc) -n 12 -d '\n' -I {} bash -c '
        run_with_logging() {
            args="$@"
            start_time=$(date +%s.%3N)
            stdbuf -oL ./build/soma $args
            exit_status=$?
            end_time=$(date +%s.%3N)
            runtime=$(awk "BEGIN {print $end_time - $start_time}")
            echo "$args | Exit: $exit_status | Time: $runtime seconds" >> "'"$log_file"'"
        }
        run_with_logging "{}"
    '
}

trap cleanup SIGINT

cleanup() {
    echo -e "\nReceived SIGINT. Cleaning up..."
    pkill -P $$
    wait
    exit 0
}

main() {
    start_time=$(date +%s.%3N)

    if [ "$eta_enabled" = true ]; then
        show_progress &
    fi

    start_parallel_tasks

    wait

    end_time=$(date +%s.%3N)

    total_runtime=$(awk "BEGIN {print $end_time - $start_time}")
    echo "Total execution time: $total_runtime seconds" >> "$log_file"
}

main

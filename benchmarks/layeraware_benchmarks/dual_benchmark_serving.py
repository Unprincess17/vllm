#!/usr/bin/env python3
"""
Script to run two benchmark_serving.py commands simultaneously.
"""

import subprocess
import threading
import sys
import time
from datetime import datetime


def run_benchmark(command, benchmark_name):
    """Run a single benchmark command and capture its output."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting {benchmark_name}...")
    
    try:
        # Run the command and capture output
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Print output in real-time with benchmark name prefix
        for line in process.stdout:
            print(f"[{benchmark_name}] {line.rstrip()}")
        
        # Wait for the process to complete
        return_code = process.wait()
        
        if return_code == 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {benchmark_name} completed successfully")
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {benchmark_name} failed with return code {return_code}")
            
        return return_code
        
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Error running {benchmark_name}: {e}")
        return 1


def main():
    # Define the two benchmark commands
    benchmark1_cmd = f"""python ../benchmark_serving_no_test.py --seed 22 --model LLM-Research/Llama-3.2-1B-Instruct --dataset-name sonnet --random-input-len 7500 --random-output-len 200 --dataset-path ../sonnet_4x.txt --sonnet-input-len 2048 --sonnet-output-len 1 --sonnet-prefix-len 50 --num-prompts 1 --burstiness 1 --request-rate 10 --port 8100 --save-result --result-dir ./results --result-filename disagg_prefill_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json"""
    
    benchmark2_cmd = f"""python ../benchmark_serving_no_test.py --seed 22 --model LLM-Research/Llama-3.2-1B-Instruct --dataset-name sonnet --random-input-len 7500 --random-output-len 200 --dataset-path ../sonnet_4x.txt --sonnet-input-len 2048 --sonnet-output-len 1 --sonnet-prefix-len 50 --num-prompts 1 --burstiness 1 --request-rate 10 --port 8200 --save-result --result-dir ./results --result-filename disagg_decode_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json"""
    
    print("=" * 80)
    print("Starting dual benchmark execution")
    print("=" * 80)
    print(f"Benchmark 1 (Port 8100): {benchmark1_cmd}")
    print(f"Benchmark 2 (Port 8200): {benchmark2_cmd}")
    print("=" * 80)
    
    # Create threads for each benchmark
    thread1 = threading.Thread(
        target=lambda: run_benchmark(benchmark1_cmd, "BENCHMARK-PREFILL"),
        name="Benchmark-prefill"
    )
    
    thread2 = threading.Thread(
        target=lambda: run_benchmark(benchmark2_cmd, "BENCHMARK-DECODE"),
        name="Benchmark-decode"
    )
    
    # Start both threads
    start_time = time.time()
    thread1.start()
    thread2.start()
    
    # Wait for both threads to complete
    thread1.join()
    thread2.join()
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    print("=" * 80)
    print(f"Both benchmarks completed in {total_duration:.2f} seconds")
    print("=" * 80)


if __name__ == "__main__":
    main()

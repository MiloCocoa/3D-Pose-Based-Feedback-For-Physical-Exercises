import os
import subprocess
import shutil
import datetime
import re

# --- Grid Search Parameters ---
# Lock in the best single values found so far
learning_rate = 0.01
lr_decay_epoch = 100
dropout_level = 0.6
weight_decay_level = 0.01 # Best value from Plan Z log

# Search over structural and decay parameters
hidden_sizes = [128, 256, 512]
num_blocks = [1, 2, 3] # Number of GC_Blocks
lr_gammas = [0.7, 0.8, 0.9] # Decay factor
# ------------------------------

base_cmd = f"python main.py --ckpt='check_point' --epoch=200 --lr={learning_rate} --lr_decay={lr_decay_epoch} --dropout={dropout_level} --weight_decay={weight_decay_level}"
log_filename = "experiment_log_broad_search.txt"

print(f"Starting broader grid search... Results will be logged to {log_filename}")

inputs_to_pipe = "Ours\ny\n"
accuracy_pattern = re.compile(r"Test_acc\(classifier\):\s*(\d+\.\d+)")

with open(log_filename, 'a', encoding='utf-8', buffering=1) as log_file:

    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file.write(f"\n===== NEW GRID SEARCH RUN (Broader Search): {start_time} =====\n")

    # Iterate through the new grid
    for hidden in hidden_sizes:
        for block in num_blocks:
            for gamma in lr_gammas:

                if os.path.isdir('Running_logs/check_point'):
                    shutil.rmtree('Running_logs/check_point')
                    print("\nCleared check_point folder.")

                # Construct the command for this specific run
                cmd = f"{base_cmd} --hidden={hidden} --block={block} --lr_gamma={gamma}"

                run_info = f"RUNNING: hidden={hidden}, block={block}, lr_gamma={gamma} (dropout={dropout_level}, wd={weight_decay_level})"

                print("---------------------------------------------------------")
                print(run_info)
                print(cmd)
                print("---------------------------------------------------------")

                log_file.write(f"\n{run_info}\n")

                result = subprocess.run(cmd,
                                        input=inputs_to_pipe,
                                        text=True,
                                        shell=True,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)

                accuracy = "N/A (Run Failed or Output Not Found)"
                if result.stdout:
                    match = accuracy_pattern.search(result.stdout)
                    if match:
                        accuracy = f"{match.group(1)}%"

                log_file.write(f"Result: Test_acc(classifier): {accuracy}\n")

print("Grid search complete.")
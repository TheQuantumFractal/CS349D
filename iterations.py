import random
import numpy as np

devices = 10000
failure_prob  = 3.7e-6 # probability of single GPU failure in 10 minute interval
recovery_time = 3 # in 10s of minutes
iteration_time = 1 # in 10s of minutes
MAX_ITERATIONS = 4000
CKPT_FREQ = 18

def simulate():
    batches_completed = 0
    state = np.ones(devices)
    failure_times = np.zeros_like(state)
    for t in range(MAX_ITERATIONS):
        state = failure_times <= 0
        failure = (np.random.rand(devices) < failure_prob)*state
        state = np.logical_and(state, 1 - failure)
        failure_times += failure*recovery_time
        failure_times -= 1
        failure_times = np.maximum(failure_times, 0)
        batches_completed += sum(state)
    return batches_completed/devices/MAX_ITERATIONS

average_ckpt_time = 0
avg_ckpt_count = 0

def simulate2():
    global average_ckpt_time
    global avg_ckpt_count
    t = 0
    batches_completed = 0
    since_last_failure = 0
    failure_count = 0
    while t < MAX_ITERATIONS:
        out = np.random.rand(1) < 1-(1-failure_prob)**devices
        if since_last_failure > CKPT_FREQ:
            since_last_failure = 0
        if sum(out) > 0:
            t += recovery_time
            t += since_last_failure
            average_ckpt_time += since_last_failure
            avg_ckpt_count += 1
            since_last_failure = 0
            failure_count += 1
        else:
            batches_completed += devices
            t += iteration_time
            since_last_failure += 1
    print(f"Number of failures in standard setting {failure_count}")
    return batches_completed/devices/MAX_ITERATIONS

print(f"Training throughput rate ours {simulate()}")
print(f"Training throughput rate standard {simulate2()}")
print(f"Average failure time between checkpoints {average_ckpt_time/avg_ckpt_count}")
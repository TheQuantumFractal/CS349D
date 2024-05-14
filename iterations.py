import random
import numpy as np

devices = 200
failure_prob  = 2e-7*100
recovery_time = 3
iteration_time = 1

def simulate():
    batches_completed = 0
    state = np.ones(devices)
    failure_times = np.zeros_like(state)
    for t in range(1000):
        state = failure_times <= 0
        failure = (np.random.rand(devices) < failure_prob)*state
        state = np.logical_and(state, 1 - failure)
        failure_times += failure*recovery_time
        failure_times -= 1
        failure_times = np.maximum(failure_times, 0)
        batches_completed += sum(state)
    print(batches_completed)

def simulate2():
    t = 0
    batches_completed = 0
    while t < 1000:
        out = np.random.rand(devices) < failure_prob
        if sum(out) > 0:
            t += recovery_time*10
        else:
            batches_completed += devices
            t += iteration_time
    return batches_completed

simulate()
s = []
for i in range(10):
    s.append(simulate2())
print(f"average: {np.mean(s)} std: {np.std(s)}")
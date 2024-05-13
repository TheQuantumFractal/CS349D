# Elastic Compute
When training on a single instance, the probability of training run failure in a day is approximately 1e-5. 
However, scaling laws necessitate exponential increases in compute. Now, if any single GPU fails, training will need to restart.
Here, the probability of failure becomes $1-(1-p)^{n}$ where $n$ is the number of GPUs and $p$ is the probability 
of failure for a single GPU. With a compute budget of 10,000 GPUs, and GPU failure probabilities of 1e-5 in a day,
the probability of failure rises to around 15%. This wastes compute due to the overhead of transferring data to and from
remote storage as well as the lost iterations between the last checkpoint. To this end, we are building an elastic training 
framework to enable large language models to never stop training due to hardware failures which we project will yield a reduction of 
15% in training GPU hours when training on 10,000 GPUs.

## TODOs
Make a separate timeout for P2P communication / heartbeats to prevent significant blocking overhead.

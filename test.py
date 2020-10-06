import numpy as np
import torch
import time

times = 100
concat_times = np.zeros(times)
preallocate_times = np.zeros(times)

for k in range(times):
    N = int(1e5)
    x = np.random.randn(N)

    y = torch.Tensor()
    start = time.time()
    for i in range(N):
        # y[i] = x[i]
        torch.cat((y,torch.Tensor([x[i]])))

    end = time.time()
    concat_times[k] = end-start




    y = torch.empty(N)
    start = time.time()
    for i in range(N):
        # y[i] = x[i]
        y[i] = x[i]

    end = time.time()
    preallocate_times[k] = end-start


print('avg concat time', np.mean(concat_times))
print('avg preallocate time', np.mean(preallocate_times))
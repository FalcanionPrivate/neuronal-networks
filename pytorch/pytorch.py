import torch
import numpy as np

print(torch.cuda.is_available())
print(torch.device("cuda"))


gpu_tensor = torch.from_numpy(np.ones(shape=(10, 10))).to(torch.device("cuda"))

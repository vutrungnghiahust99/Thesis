import torch
import random
from tqdm import tqdm

train = torch.load('data/mnist/MNIST/processed/training.pt')

N = 60000

m = [[] for _ in range(N)]
m1 = []

for i in tqdm(range(100)):
    for _ in range(N):
        a = random.randint(0, 59999)
        m[a].append(i)


for i in range(N):
    b = ','.join([str(x) for x in m[i]])
    m1.append(b)

d = (train[0], train[1], m1)

torch.save(d, 'training_rf.pt')

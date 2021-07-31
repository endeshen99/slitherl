import torch

from slitherl_env import SlitherlEnv
import time

from torch.nn import functional as F

EPS = 1e-6




a = torch.randn(4).unsqueeze(-1)
b = torch.ones(4, 2)
print(a)
print(a*b)



env_num = 4
snake_num = 10

env = SlitherlEnv(env_num, snake_num)

for _ in range(300):
    #env.primitive_render()
    env.render()
    # actions: 0 forward, -1 counterclockwise, 1 clockwise
    actions = torch.randint(-1, 2, (env_num, snake_num)).float()
    env.step(actions)
    time.sleep(0.1)
    print("good")









"""


ORIENTATION_FILTERS = torch.tensor([
    [
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 0],
    ],
    [
        [0, 0, 0],
        [0, -1, 1],
        [0, 0, 0],
    ],
    [
        [0, 0, 0],
        [0, -1, 0],
        [0, 1, 0],
    ],
    [
        [0, 0, 0],
        [1, -1, 0],
        [0, 0, 0],
    ],
]).unsqueeze(1).float()


a = torch.tensor([[1,0], [2, 0]]) <EPS
print(a)
print(torch.ones(2,2).add_(a))



actions = torch.zeros(2,2)
orientations = torch.ones(2,2)
orientations.add_(actions)
actions_onehot = torch.zeros(2,2,4)
actions_onehot.scatter_(-1, orientations.unsqueeze(-1).long(), 1)
actions_onehot_shape = actions_onehot.view(4, 4)
#print(actions_onehot)

snakes = torch.zeros(2, 3, 3)
snakes[0, 1, 1] = 1
snakes = snakes#.unsqueeze(0).unsqueeze(0)
snakes = snakes.repeat(2, 2, 1, 1, 1)
#print(snakes)


heads = snakes[:,:, 0:1, :,:].view(4, 1, 3, 3)
head_deltas = F.conv2d(heads, ORIENTATION_FILTERS, padding = 1)
#head_deltas should be shape (env_num*snake_num) * 4 *size * size
head_deltas = torch.einsum('bchw,bc->bhw', [head_deltas, actions_onehot_shape]).unsqueeze(1)
heads.add_(head_deltas)

#print(snakes)


heads = snakes[:,:, 0:1, :,:].view(4, 1, 3, 3)
head_deltas = F.conv2d(heads, ORIENTATION_FILTERS, padding = 1)
#head_deltas should be shape (env_num*snake_num) * 4 *size * size
head_deltas = torch.einsum('bchw,bc->bhw', [head_deltas, actions_onehot_shape]).unsqueeze(1)
heads.add_(head_deltas)

#print(snakes)



a = torch.zeros(2, 2,4)
c = a.view(4, 4)
c[0,0] = 1
#print(c)
#print(a)
b = torch.tensor([
    [0,1],
    [2, 3]
    ]).unsqueeze(-1)
a.scatter_(-1, b, 1)
#print(a)

a[a!=0] = 2
print(a.size() == (2,2,4))

a = torch.sum(torch.sum(a, 1), 1)
print(a.size())
print(a)





head = torch.zeros(5, 1, 3, 3)
head[:,0:1, 1, 1].add_(torch.ones(5,1))
print(head)

x = head.repeat(1,2,1,1)
print(x.size())


filter = torch.tensor([
    [
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 0],
    ],
    [
        [0, 0, 0],
        [0, -1, 1],
        [0, 0, 0],
    ]
    ]).unsqueeze(0).float()

filters = torch.cat(())

print(F.conv2d(head, filter, padding = 1))



a = torch.ones(3,3)
b = torch.randn(3,3)
print(a)
print(a.sum())
print(b)
print(a.byte())



z = torch.zeros(3,3,3)

b = torch.ones(3,1,3)
z[:, 0:1].add_(b)
print(z)


z = torch.zeros(3, 2,2)
z[0,0,0] = 1
z[1, 1,1] = 1
z[2, 0,1] = 1
weights = torch.tensor([1,2,0])
output = torch.einsum('bhw,b->hw', z.float(), weights.float())
print(z)
print(-1*output)


head_idx = ten[:, 0, :, :].view(3, 4 ** 2).argmax(dim=-1)

print(head_idx)

observation = torch.Tensor([
                head_idx // 4,
                head_idx % 4,
            ])
print(observation)

"""
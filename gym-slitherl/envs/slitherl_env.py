import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from gym.spaces.discrete import Discrete
from gym.envs.classic_control import rendering
import torch
from torch.nn import functional as F



EPS = 1e-6


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



class SlitherlEnv(gym.Env):
  metadata = {'render.modes': ['human']}
  def __init__(self, env_num, snake_num, size = 40):
    self.size = size
    self.snake_num = snake_num
    self.env_num = env_num
    self.snakes = torch.zeros(env_num, snake_num, 2, size, size)
    self.fruits = torch.zeros(env_num, 1, size, size)
    #initiating the orientation of the snakes, 0, 1,2,3 are the four directions, zero is down, 1 is left
    self.orientations = torch.ones(env_num, snake_num)

    #putting down snake_num many length zero snake heads
    for i in range(snake_num):
      self.snakes[:, i, 0, 4*i, 4*i].add_(torch.ones(env_num))
      self.snakes[:, i, 1, 4*i, 4*i + 1].add_(2*torch.ones(env_num))
      self.snakes[:, i, 1, 4*i, 4*i + 2].add_(torch.ones(env_num))
    
    self.snake_viewer = rendering.Viewer(size*10, size*10)
    self.snake_viewer.render(return_rgb_array=True)

  



  #action is of size env_num*snake_num, can be -1, 0, 1
  def _move_head_body(self, actions):
    #first, we move the head channel to the new positions
    #print(self.orientations.size())
    #print(actions.size())
    self.orientations.add_(actions).add_(4 * torch.ones(self.env_num, self.snake_num)).fmod_(4)
    actions_onehot = torch.zeros(self.env_num, self.snake_num, 4)
    #print(self.orientations)
    actions_onehot.scatter_(-1, self.orientations.unsqueeze(-1).long(), 1)
    actions_onehot_shape = actions_onehot.view(self.env_num*self.snake_num, 4)
    #here, actions_onehot should be of size env_num*snake_num*4
    heads = self.snakes[:,:, 0:1, :,:].view(self.env_num*self.snake_num, 1, self.size, self.size)
    head_deltas = F.conv2d(heads, ORIENTATION_FILTERS, padding = 1)
    #head_deltas should be shape (env_num*snake_num) * 4 *size * size
    head_deltas = torch.einsum('bchw,bc->bhw', [head_deltas, actions_onehot_shape]).unsqueeze(1)
    heads.add_(head_deltas)

    #Now, we determine whether fruit is eaten and update the body
    fruits = self.fruits.repeat(1, self.snake_num, 1, 1)
    hit_fruit = (self.snakes[:, :, 0, :, :] * fruits).sum(-1).sum(-1)
    assert hit_fruit.size() == (self.env_num, self.snake_num)
    hit_fruit[hit_fruit!=0] = 1
    hit_fruit.add_(-torch.ones(self.env_num, self.snake_num))
    body_deltas = hit_fruit.unsqueeze(-1).unsqueeze(-1).expand(self.env_num, self.snake_num, self.size, self.size)
    #at this point, hit_fruit = 0 means that fuit is eaten, -1 means no fruit
    #print(self.snakes[:,:, 1, :, :].size())
    #print(body_deltas.size())
    self.snakes[:, :, 1, :, :] = self.snakes[:,:, 1, :, :].add_(body_deltas).relu()
    #now we add the neck of the snake
    previous_heads = (heads - head_deltas).view(self.env_num, self.snake_num, 1, self.size, self.size)
    snake_sizes = torch.max(self.snakes[:,:, 1,:,:].view(self.env_num, self.snake_num, self.size * self.size), -1)[0] \
      + torch.ones(self.env_num, self.snake_num)
    assert snake_sizes.size() == (self.env_num, self.snake_num)
    snake_sizes = snake_sizes.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(self.env_num, self.snake_num, 1, self.size, self.size)
    self.snakes[:,:, 1:2, :,:].add_(previous_heads * snake_sizes)

    #finally, we delete the fruit that was eaten.
    all_heads = self.snakes[:,:, 0, :,:].sum(1).unsqueeze(1)
    assert all_heads.size() == (self.env_num, 1, self.size, self.size)
    self.fruits = self.fruits.add_(-all_heads).relu()
    


  #sum up all the snakes in the same environment, size (env_num, size, size)
  def _snake_pos(self):
    return self.snakes.sum(1).sum(1).round()



  #this function spawn fruits when there are no fruit available.
  def _spawn_fruit(self):
    coord = torch.randint(0, self.size, (2,))
    new_fruit = torch.zeros(self.env_num, 1, self.size, self.size)
    new_fruit[:, 0 , coord[0], coord[1]].add_(torch.ones(self.env_num))
    #making sure nothing is added if it is a snake position
    snake_positions = self._snake_pos().unsqueeze(1)
    new_fruit = new_fruit.add_(-snake_positions).relu()
    #determine if the environment is empty of food.
    have_fruit = self.fruits.sum(-1).sum(-1)
    have_fruit[have_fruit != 0] = -1
    have_fruit = have_fruit.add_(torch.ones(self.env_num, 1)).unsqueeze(-1).unsqueeze(-1) #now being 1 iff not have food
    have_fruit = have_fruit.expand(self.env_num, 1, self.size, self.size)
    new_fruit = new_fruit * have_fruit
    self.fruits.add_(new_fruit)
    
  
  def _collisions(self):
    #here, we will create a tensor kill of size env_num, snake_num. equals 1 if that snake is alive, 0 otherwise
    kill = torch.ones(self.env_num, self.snake_num)
    #first we look at those that have their head at the boundary, i.e. head channel =0
    boundary_snakes = self.snakes[:, :, 0, :, :].sum(-1).sum(-1) < EPS
    kill.add_(-(boundary_snakes.int()))
    #now we look at those that collide with other snakes

    #now we add the fruits coming from the corpse of snakes

    kill = kill.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(self.env_num, self.snake_num, 2, self.size, self.size)
    self.snakes = self.snakes * kill

    


  def step(self, action):
    self._move_head_body(action)
    self._collisions()
    self._spawn_fruit()
  
    #print("step all good")
    


  def reset(self):
    ...
  def render(self, mode='human'):
    self.snake_viewer.geoms.clear()
    fruit_and_snake = self._snake_pos() + self.fruits[:, 0, :, :]
    for i in range(self.size):
      for j in range(self.size):
        if fruit_and_snake[0,i,j] > EPS:
          segment = rendering.FilledPolygon([(10*i, 10*j + 10), 
                                    (10*i, 10*j), 
                                    (10*i + 10, 10*j), 
                                    (10*i + 10, 10*j + 10)])
          self.snake_viewer.add_geom(segment)
    self.snake_viewer.render(return_rgb_array=True)


  def close(self):
    ...
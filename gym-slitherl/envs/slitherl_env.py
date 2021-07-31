import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from gym.spaces.discrete import Discrete
from gym.envs.classic_control import rendering
import torch
from torch.nn import functional as F
from PIL import Image


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
  def __init__(self, env_num, snake_num, size = 40, resize_scale = 5):
    self.size = size
    self.resize_scale = resize_scale
    self.snake_num = snake_num
    self.env_num = env_num
    self.snakes = torch.zeros(env_num, snake_num, 2, size, size)
    self.fruits = torch.zeros(env_num, size, size)
    self.reward = torch.zeros(env_num, snake_num)
    #initiating the orientation of the snakes, 0, 1,2,3 are the four directions, zero is down, 1 is left
    self.orientations = torch.ones(env_num, snake_num)

    #putting down snake_num many length zero snake heads
    for i in range(snake_num):
      self.snakes[:, i, 0, 4*i, 4*i].add_(torch.ones(env_num))
      self.snakes[:, i, 1, 4*i, 4*i + 1].add_(2*torch.ones(env_num))
      self.snakes[:, i, 1, 4*i, 4*i + 2].add_(torch.ones(env_num))
    
    self.viewers = [rendering.SimpleImageViewer() for _ in range(1)]

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
    eaten_fruit_snakes = (self.snakes[:, :, 0, :, :] * self.fruits[:, None, :, :]).sum(-1).sum(-1)
    assert eaten_fruit_snakes.size() == (self.env_num, self.snake_num)
    self.reward.add_((eaten_fruit_snakes > EPS).float())
    body_deltas = (eaten_fruit_snakes > EPS).float() - 1
    #at this point, body_deltas = 0 means that fruit is eaten, -1 means no fruit
    self.snakes[:, :, 1, :, :] = self.snakes[:,:, 1, :, :].add_(body_deltas[..., None, None]).relu()
    #now we add the neck of the snake
    previous_heads = (heads - head_deltas).view(self.env_num, self.snake_num, 1, self.size, self.size)
    snake_sizes = torch.max(self.snakes[:,:, 1,:,:].view(self.env_num, self.snake_num, self.size * self.size), -1)[0] \
      + torch.ones(self.env_num, self.snake_num)
    assert snake_sizes.size() == (self.env_num, self.snake_num)
    snake_sizes = snake_sizes.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(self.env_num, self.snake_num, 1, self.size, self.size)
    self.snakes[:,:, 1:2, :,:].add_(previous_heads * snake_sizes)

    #finally, we delete the fruit that was eaten.
    all_heads = self.snakes[:,:, 0, :,:].sum(1)
    assert all_heads.size() == (self.env_num, self.size, self.size)
    self.fruits = self.fruits.add_(-all_heads).relu()
    


  #sum up all the snakes in the same environment, size (env_num, size, size)
  def _snake_pos(self):
    return self.snakes.sum(1).sum(1).round()



  #this function spawn fruits when there are no fruit available.
  def _spawn_fruit(self):
    coord = torch.randint(0, self.size, (2,))
    new_fruit = torch.zeros(self.env_num, self.size, self.size)
    new_fruit[:, coord[0], coord[1]].add_(torch.ones(self.env_num))
    #making sure nothing is added if it is a snake position
    snake_positions = self._snake_pos()
    new_fruit = new_fruit.add_(-snake_positions).relu()
    #determine if the environment is empty of food.
    have_fruit = self.fruits.sum(-1).sum(-1) < EPS
    new_fruit = new_fruit * have_fruit[..., None, None].float()
    self.fruits.add_(new_fruit)
    
  
  def _collisions(self):
    #here, we will create a tensor preserve of size env_num, snake_num. equals 1 if that snake is alive, 0 otherwise
    preserve = torch.ones(self.env_num, self.snake_num)
    #first we look at those that have their head at the boundary, i.e. head channel =0
    boundary_snakes = self.snakes[:, :, 0, :, :].sum(-1).sum(-1) < EPS
    assert boundary_snakes.size() == (self.env_num, self.snake_num)
    preserve.add_(-(boundary_snakes.float()))
    #now we look at those that collide with other snakes or with themselves
    #first check if collided into any snake's body
    body_collided_snakes = (self.snakes[:, :, 1, :, :].sum(1).unsqueeze(1) * self.snakes[:, :, 0, :, :]).sum(-1).sum(-1) > EPS
    assert body_collided_snakes.size() == (self.env_num, self.snake_num)
    preserve.add_(-(body_collided_snakes.float()))
    #then check if collided into other snake's head
    head_collided_snakes = (self.snakes[:, :, 0, :, :].sum(1).unsqueeze(1) - self.snakes[:, :, 0, :, :]).prod(-1).prod(-1) > EPS
    assert head_collided_snakes.size() == (self.env_num, self.snake_num)
    preserve.add_(-(head_collided_snakes.float()))
    
    #now we add the fruits coming from the corpse of snakes
    kill = (boundary_snakes + body_collided_snakes + head_collided_snakes).float()
    assert kill.size() == (self.env_num, self.snake_num)
    new_fruit = (self.snakes * kill[..., None, None, None]).sum(1).sum(1)
    assert new_fruit.size() == (self.env_num, self.size, self.size)
    self.fruits.add_(new_fruit)
    self.reward.add_((self.reward >= 0).float() * (kill * -100.0))

    #update the snakes
    self.snakes = self.snakes * preserve[..., None, None, None]

    ## optional logging ##
    # if boundary_snakes.sum() > 0:
    #   print("hit boundary")
    # if body_collided_snakes.sum() > 0:
    #   print("hit body")
    # if head_collided_snakes.sum() > 0:
    #   print("hit head")

    # print ("the number of killed snakes:")
    # print((kill.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).sum()))
    new_fruit = (self.snakes[:, :, 1: 2, :, :] * kill.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)).sum(1).sum(1).unsqueeze(1)
    self.fruits.add_(new_fruit)

    preserve = preserve.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(self.env_num, self.snake_num, 2, self.size, self.size)
    self.snakes = self.snakes * preserve

    


  def step(self, action):
    self._move_head_body(action)
    self._collisions()
    self._reset_dead_env()
    self._spawn_fruit()

    #print(self.snakes[0,0,1,:,:])
  
    # print(self.reward)
    #print("step all good")
    
  def _reset_dead_env(self):
    new_snakes = torch.zeros(self.env_num, self.snake_num, 2, self.size, self.size)
    for i in range(self.snake_num):
      new_snakes[:, i, 0, 4*i, 4*i].add_(torch.ones(self.env_num))
      new_snakes[:, i, 1, 4*i, 4*i + 1].add_(2*torch.ones(self.env_num))
      new_snakes[:, i, 1, 4*i, 4*i + 2].add_(torch.ones(self.env_num))
    need_reset = self.snakes[:, :, 0, :,:].sum(1).sum(-1).sum(-1) < EPS
    need_reset = need_reset.int()
    new_snakes = new_snakes * need_reset.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    self.snakes = self.snakes + new_snakes
    perserve_fruit = - (need_reset - torch.ones(self.env_num))
    self.fruits = self.fruits * (perserve_fruit.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))










  def reset(self):
    ...

  

  def primitive_render(self, mode='human'):
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


  def render(self, mode='human'):

    head_color = [0, 0, 0]
    body_color = [105, 105, 105]
    board_color = [255, 255, 255]
    fruit_color = [255, 0, 0]

    for idx, viewer in enumerate(self.viewers):   # rendering one env at a time
      snakes = self.snakes[0:1,:,:,:,:].detach().cpu().numpy()
      fruits = self.fruits[0:1,:,:,:].detach().cpu().numpy()

      # fruit_pos shape is size by size
      fruit_pos = fruits[idx, :, :]
      # stack three times to generate rgb array
      fruit_rgb = np.stack([fruit_pos, fruit_pos, fruit_pos], axis = -1).astype(np.uint8) * fruit_color

      # head_pos shape is size by size
      head_pos = (snakes[idx, :, 0, :, :].sum(axis = 0) > 0).astype(np.uint8)
      head_rgb = np.stack([head_pos, head_pos, head_pos], axis = -1) * head_color

      # body_pos shape is size by size
      body_pos = (snakes[idx, :, 1, :, :].sum(axis = 0) > 0).astype(np.uint8)
      body_rgb = np.stack([body_pos, body_pos, body_pos], axis = -1) * body_color

      # board_pos are where there are no heads, fruits, or bodies
      board_pos = (snakes[idx, :, 0, :, :].sum(axis = 0) + snakes[idx, :, 1, :, :].sum(axis = 0)+ fruit_pos <= 0).astype(np.uint8)
      board_rgb = np.stack([board_pos, board_pos, board_pos], axis = -1) * board_color

      # add the rgb arrays to generate the whole image
      img_to_show = head_rgb + board_rgb + body_rgb + fruit_rgb

      # enlarge the tensor to become an actual board/image
      img_to_show_enlarged = np.array(Image.fromarray(img_to_show.astype(np.uint8)).resize(
        (self.size * self.resize_scale,
        self.size * self.resize_scale), resample=Image.NEAREST
      ))

      viewer.imshow(img_to_show_enlarged)


  def close(self):
    ...
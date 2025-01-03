import numpy as np
from tinygrad import Tensor
import matplotlib.pyplot as plt
from dataclasses import dataclass
import time, random, sys, os, uuid, json
from pathlib import Path
plt.ion()  # Turn on interactive mode

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
MOVES = [UP, DOWN, LEFT, RIGHT]

@dataclass
class State:
  board: np.ndarray|None
  move: tuple[int, int]|None
  snake: list[tuple[int, int]]|None
  head: tuple[int, int]|None
  food: tuple[int, int]|None
  size: tuple[int, int]|None
  score: int = 0

sample_board = np.array([
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 3, 2, 2, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
], dtype=np.int8)

sample_state = State(board=sample_board, move=LEFT, snake=[(2, 4), (2, 5), (2, 6)], head=(2, 4), food=(5,4), size=(10,10))

def show_board(state: State, sleep=1):
  board = state.board
  # Clear previous plots
  plt.clf()
  
  # Create a colormap
  cmap = plt.cm.colors.ListedColormap(['white', 'red', 'blue', 'darkblue'])
  
  # Plot the board using numerical values directly
  plt.imshow(board, cmap=cmap, aspect='equal', vmin=0, vmax=3)
  
  # Add grid
  plt.grid(False)  # Turn off default grid
  
  # Add lines for the grid that match cell edges
  for x in range(len(board[0]) + 1):
      plt.axvline(x - 0.5, color='black', linewidth=1)
  for y in range(len(board) + 1):
      plt.axhline(y - 0.5, color='black', linewidth=1)
  
  # Adjust ticks to match cell centers
  # plt.xticks(range(len(board[0])))
  # plt.yticks(range(len(board)))
  plt.axis('off')

  # Score
  plt.text(state.size[1]/2, -0.5, f'Score: {state.score}', 
          horizontalalignment='center', 
          fontsize=12)

  # Update the display without blocking
  # plt.show(block=False)
  # plt.pause(0.001)  # Very small pause to ensure display updates
  plt.draw()
  plt.pause(sleep)

def make_move(state:State, move=None):
  if move is not None: state.move = move
  else: move = state.move
  new_head = (state.head[0] + move[0], state.head[1] + move[1])
  state.head = new_head
  if new_head == state.food:
    state.score+=1
    state.snake = [new_head] + state.snake
    place_food(state)
  else: state.snake = [new_head] + state.snake[0:-1]
  rebuild_board(state)

def place_food(state: State):
  new_food = (np.random.randint(0, state.size[0]), np.random.randint(0, state.size[1]))
  while new_food in state.snake:
    new_food = (np.random.randint(0, state.size[0]), np.random.randint(0, state.size[1]))
  state.food = new_food

def rebuild_board(state: State):
  new_board = np.zeros(state.size,np.int8)
  new_board[state.food] = 1
  new_board[state.head] = 3
  for s in state.snake[1:]:
    new_board[s] = 2
  state.board = new_board
  

def get_head(board): return np.column_stack(np.where(board == 3)).reshape(-1)

def get_move_options(state: State):
  opts = [False, False, False, False]
  for i, m in enumerate(MOVES):
    new_head = (state.head[0] + m[0], state.head[1] + m[1])
    if new_head in state.snake: continue
    if new_head[0]>=0 and new_head[0]<state.size[0] and new_head[1]>=0 and new_head[1]<state.size[1]: opts[i]=True

  # return opts
  return {k: v for k, v in zip(MOVES, opts)}

def random_valid_move(state: State):
  moves = get_move_options(state)
  valid_moves = [move for move, valid in moves.items() if valid]
  # TODO: Make better
  # return random.choice(valid_moves) if valid_moves else sys.exit()
  return random.choice(valid_moves) if valid_moves else None

def build_start_board(size=(10, 10)):
  head = (size[0]//2, size[1]//2)
  snake = [head]
  for i in range(2):
    snake.append((snake[-1][0], snake[-1][1]+1))
  # print(snake)
  state = State(None, None, snake, snake[0], None, size)
  place_food(state)
  rebuild_board(state)
  return state
# print(build_start_board())

def capture_state(state: State, game_id, move_id, subdir='data'):
  path = Path.cwd()/subdir/game_id
  path.mkdir(exist_ok=True, parents=True)
  
  # np.savez_compressed(path/f'{move_id}.npz', board=state.board)
  np.save(path/f'{move_id}.npy', state.board)

  state_dict = {
        'move': state.move,
        'snake': state.snake,
        'head': state.head,
        'food': state.food,
        'size': state.size,
        'score': state.score
    }
  with open(path/f'{move_id}.json', 'w') as f:
    json.dump(state_dict, f)

  # with open(filename, 'wb') as f:
  #   pickle.dump(obj, f)

# capture_state(sample_state, 'test_sample', 1)

def load_state(game_id, move_id, subdir='data') -> State:
    path = Path.cwd()/subdir/game_id
    # board = np.load(path/f'{move_id}.npz')['board']
    board = np.load(path/f'{move_id}.npy')

    with open(path/f'{move_id}.json', 'r') as f:
        state_dict = json.load(f)

    return State(
        board=board,
        **state_dict
    )

# print(load_state('test_sample', '1'))

# TODO: WIP
# Make dynamic snake gen
# def build_random_board(size=(10, 10), snake_len=5):

def play_game(subdir='data', show_game=False, move_func=random_valid_move, log=False, save_game=False):
  state = build_start_board()
  game_over = False
  game_id = uuid.uuid4().__str__()
  move_id = 0
  if save_game: capture_state(state, game_id, move_id, subdir)
  stats = {'game_id':game_id, 'step_times':[]}
  while not game_over:
    # print('MOVE', state.move)
    s = time.perf_counter()
    if show_game: show_board(state, .001)
    move = move_func(state)
    # MUDDY logic with move being none for capturing state....
    state.move = move
    if move is None: game_over = True
    else: make_move(state, move)
    move_id+=1
    e = time.perf_counter()
    if save_game: capture_state(state, game_id, move_id, subdir)
    if log: stats['step_times'].append(e-s)
  if log: stats['move_cnt'] = move_id
  if log: stats['score'] = state.score
  return stats

# GAME LOOP
game_cnt = 0
game_times = []
scores = []
while game_cnt<100:
  # print('PLAYING GAME:', game_cnt)
  game_cnt+=1
  stats = play_game('data/test', log=True, save_game=False)
  game_times = game_times+stats['step_times']
  scores.append(stats['score'])
print('STEPS/SEC:', 1/np.array(game_times).mean())
print('SCORE:', np.array(scores).mean())

# print('TIME/step:', np.array(game_times).mean())


# while True:
#   state = build_start_board()
#   game_over = False
#   game_id = uuid.uuid4().__str__()
#   move_id = 0
#   capture_state(state, game_id, move_id, 'data/test')
#   while not game_over:
#     # print('MOVE', state.move)

#     show_board(state, .001)
#     move = random_valid_move(state)
#     # MUDDY logic with move being none for capturing state....
#     state.move = move
#     if move is None: game_over = True
#     else: make_move(state, move)
#     move_id+=1
#     capture_state(state, game_id, move_id, 'data/test')

#   print('FINAL SCROE:', state.score)


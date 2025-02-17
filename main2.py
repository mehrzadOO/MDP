# Import nessary libraries
import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.cliffwalking import CliffWalkingEnv
from gymnasium.error import DependencyNotInstalled
from os import path


# Do not change this class
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
image_path = path.join(path.dirname(gym.__file__), "envs", "toy_text")

class CliffWalking(CliffWalkingEnv):
    def __init__(self, is_hardmode=True, num_cliffs=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_hardmode = is_hardmode

        # Generate random cliff positions
        if self.is_hardmode:
            self.num_cliffs = num_cliffs
            self._cliff = np.zeros(self.shape, dtype=bool)
            self.start_state = (3, 0)
            self.terminal_state = (self.shape[0] - 1, self.shape[1] - 1)
            self.cliff_positions = []
            while len(self.cliff_positions) < self.num_cliffs:
                new_row = np.random.randint(0, 4)
                new_col = np.random.randint(0, 11)
                state = (new_row, new_col)
                if (
                    (state not in self.cliff_positions)
                    and (state != self.start_state)
                    and (state != self.terminal_state)
                ):
                    self._cliff[new_row, new_col] = True
                    if not self.is_valid():
                        self._cliff[new_row, new_col] = False
                        continue
                    self.cliff_positions.append(state)

        # Calculate transition probabilities and rewards
        self.P = {}
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            self.P[s] = {a: [] for a in range(self.nA)}
            self.P[s][UP] = self._calculate_transition_prob(position, [-1, 0])
            self.P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1])
            self.P[s][DOWN] = self._calculate_transition_prob(position, [1, 0])
            self.P[s][LEFT] = self._calculate_transition_prob(position, [0, -1])

    def _calculate_transition_prob(self, current, delta):
        new_position = np.array(current) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        if self._cliff[tuple(new_position)]:
            return [(1.0, self.start_state_index, -100, False)]

        terminal_state = (self.shape[0] - 1, self.shape[1] - 1)
        is_terminated = tuple(new_position) == terminal_state
        return [(1 / 3, new_state, -1, is_terminated)]

    # DFS to check that it's a valid path.
    def is_valid(self):
        frontier, discovered = [], set()
        frontier.append((3, 0))
        while frontier:
            r, c = frontier.pop()
            if not (r, c) in discovered:
                discovered.add((r, c))
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                for x, y in directions:
                    r_new = r + x
                    c_new = c + y
                    if r_new < 0 or r_new >= self.shape[0] or c_new < 0 or c_new >= self.shape[1]:
                        continue
                    if (r_new, c_new) == self.terminal_state:
                        return True
                    if not self._cliff[r_new][c_new]:
                        frontier.append((r_new, c_new))
        return False

    def step(self, action):
        if action not in [0, 1, 2, 3]:
            raise ValueError(f"Invalid action {action}   must be in [0, 1, 2, 3]")

        if self.is_hardmode:
                if action == 0:
                    action = np.random.choice([0, 1, 3], p=[1 / 3, 1 / 3, 1 / 3])
                elif action == 1:
                    action = np.random.choice([0, 1, 2], p=[1 / 3, 1 / 3, 1 / 3])
                elif action == 2:
                    action = np.random.choice([1, 2, 3], p=[1 / 3, 1 / 3, 1 / 3])
                elif action == 3:
                    action = np.random.choice([0, 2, 3], p=[1 / 3, 1 / 3, 1 / 3])

        return super().step(action)

    def _render_gui(self, mode):
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[toy-text]`"
            ) from e
        if self.window_surface is None:
            pygame.init()

            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("CliffWalking")
                self.window_surface = pygame.display.set_mode(self.window_size)
            else:  # rgb_array
                self.window_surface = pygame.Surface(self.window_size)
        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.elf_images is None:
            hikers = [
                path.join(image_path, "img/elf_up.png"),
                path.join(image_path, "img/elf_right.png"),
                path.join(image_path, "img/elf_down.png"),
                path.join(image_path, "img/elf_left.png"),
            ]
            self.elf_images = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in hikers
            ]
        if self.start_img is None:
            file_name = path.join(image_path, "img/stool.png")
            self.start_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.goal_img is None:
            file_name = path.join(image_path, "img/cookie.png")
            self.goal_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.mountain_bg_img is None:
            bg_imgs = [
                path.join(image_path, "img/mountain_bg1.png"),
                path.join(image_path, "img/mountain_bg2.png"),
            ]
            self.mountain_bg_img = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in bg_imgs
            ]
        if self.near_cliff_img is None:
            near_cliff_imgs = [
                path.join(image_path, "img/mountain_near-cliff1.png"),
                path.join(image_path, "img/mountain_near-cliff2.png"),
            ]
            self.near_cliff_img = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in near_cliff_imgs
            ]
        if self.cliff_img is None:
            file_name = path.join(image_path, "img/mountain_cliff.png")
            self.cliff_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )

        for s in range(self.nS):
            row, col = np.unravel_index(s, self.shape)
            pos = (col * self.cell_size[0], row * self.cell_size[1])
            check_board_mask = row % 2 ^ col % 2
            self.window_surface.blit(self.mountain_bg_img[check_board_mask], pos)

            if self._cliff[row, col]:
                self.window_surface.blit(self.cliff_img, pos)
            if s == self.start_state_index:
                self.window_surface.blit(self.start_img, pos)
            if s == self.nS - 1:
                self.window_surface.blit(self.goal_img, pos)
            if s == self.s:
                elf_pos = (pos[0], pos[1] - 0.1 * self.cell_size[1])
                last_action = self.lastaction if self.lastaction is not None else 2
                self.window_surface.blit(self.elf_images[last_action], elf_pos)

        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )
#----------------------------------------------------

def get_neighbors(action):
    if action == 0:
        return (0, 1, 3)
    elif action == 1:
        return (1, 0, 2)
    elif action == 2:
        return (2, 1, 3)
    elif action == 3:
        return (3, 0, 2)

def set_rewards():
    rewards = {}
    for state in range(48):
        column = state % 12
        row = int(state / 12)
        dist_row = abs(row - 3)
        dist_column = abs(column - 11)
        if (column >= 0) and (column <= 7):
            rewards[state] = - 0.3 * (dist_row + dist_column)
        else:
            rewards[state] = - 0.2 * (dist_row + dist_column)

    return rewards

def value_iteration(cliff_positions , rewards):
    # initalize
    V = {}
    for state in range(48):
        V[state] = 0.

    def possible_actions(state):
        possible_actions = []
        for action in range(4):

            if not (action == 3 and state == 0):
                    possible_actions.append(action)

        return possible_actions

    def possible_actions2(state):
        possible_actions = []
        for action in range(4):
            next_state = step(state, action)

            left_side = state % 12
            right_side = (state % 12) - 11

            if not (action == 3 and left_side == 0) and not (action == 1 and right_side == 0):
                if next_state >= 0 and next_state < 48:
                    possible_actions.append(action)
        return possible_actions
    def step(state, action):
        if action == 0:
            return state - 12
        elif action == 1:
            return state + 1
        elif action == 2:
            return state + 12
        elif action == 3:
            return state - 1


    def Q(state, action):
        sum = 0
        flag = False
        left_side = state % 12
        right_side = (state % 12) - 11
        for possible_action in get_neighbors(action):

            next_state = step(state, possible_action)
            if (possible_action == 3 and left_side == 0) or (possible_action == 1 and right_side == 0) or \
                    next_state < 0 or next_state > 47:
                sum += -1
                continue
            if next_state in cliff_positions:
                reward = -50
                prob = float(1/3)
            elif next_state == 47:  #End State
                reward = 0
                prob = 1/3
                flag = True
            else:
                reward = rewards[next_state]
                prob = float(1/3)
                flag = True

            q = prob * (float(reward) + 0.9 * V[next_state])
            sum += q

        return sum
        # next_state, reward, done, truncated, info = env.step(action)
        # return float(info['prob']) * (float(reward) + 0.9 * V[next_state])

    pi = {}

    while True:
        newV = {}
        for state in range(48):
            if state == 47:
                newV[state] = 0
            elif state in cliff_positions:
                newV[state] = -50
            else:
                newV[state] = max(Q(state, action) for action in possible_actions(state))

        # checking for convergence
        if max(abs(V[state] - newV[state]) for state in range(48)) < 1e-10 :
            break
        V = newV

        # read out the policy
        pi = {}
        for state in range(48):
            if state == 47:
                pi[state] = 'none'
            else:
                # l = []
                # for action in possible_actions(state):
                #     l.append((Q(state, action), action))
                # l.sort()
                # pi[state] = l[len(l)-1][1]
                pi[state] =max((Q(state, action), action) for action in possible_actions(state))[1]
        # print(cliff_positions)
        # print(pi)
        # print('#############')

    return pi, V

#----------------------------------------------------




# Create an environment
env = CliffWalking(render_mode="human")
observation, info = env.reset(seed=30)

cliff_positions = []

for cliff_pos in env.cliff_positions:
    cliff_positions.append(cliff_pos[0] * 12 + cliff_pos[1])

rewards = set_rewards()

pi , v = value_iteration(cliff_positions, rewards)
print(f'pi : {pi}\ncliff_positions : {cliff_positions}\nrewards : {rewards}\n values : {v}')
# Define the maximum number of iterations
max_iter_number = 1000

i = 0
for __ in range(max_iter_number):
        # TODO: Implement the agent policy here
    # Note: .sample() is used to sample random action from the environment's action space

    # Choose an action (Replace this random action with your agent's policy)
    # action = env.action_space.sample()

    # print(env.s)
# Perform the action and receive feedback from the environment
    action = pi[env.s]
    state = env.s
    next_state, reward, done, truncated, info = env.step(action)

    print(f'action : {action}, state : {state} , next_state : {next_state}, reward : {reward}')
    if done or truncated:
        observation, info = env.reset()
        if done :
            i+=1
print(f'i is {i}')

# Close the environment
env.close()










import numpy as np
import gymnasium as gym
from os import path
from gymnasium.envs.toy_text import FrozenLakeEnv
import signal
import sys
import pygame

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
image_path = path.join(path.dirname(gym.__file__), "envs", "toy_text")
start_position=0
Tname = None
pth = None
def saveST():
	global start_position,pth
	if pth:
		position = pygame.mixer.music.get_pos() // 1000 + start_position
		with open(pth, 'w') as f:
			f.write(str(position))
		pygame.mixer.music.stop()
		pygame.mixer.quit()

def Interrupt(sig, frame):
	saveST()
	sys.exit()

class FrozenLake(FrozenLakeEnv):
	def __init__(self, is_hardmode=True, numhole_positions=10, *args, **kwargs):
		if len(sys.argv) == 2:
			arg = sys.argv[1]
			if arg.startswith("soundtrack="):
				Arg = arg.split("=")[1]
				if Arg != '0':
					Path = 'files/'+Arg
					if path.exists(Path):
						global Tname, pth
						Tname = Arg
						pth = 'files/'+Tname+'.txt'
						self.DJ(Path)
						signal.signal(signal.SIGINT, Interrupt)
					else: print('Track not found!')
		super().__init__(*args, **kwargs)
		self.is_hardmode = is_hardmode
		self.shape = (8, 8)  # 8x8 frozen lake
		
		self.start_state = (0, 0)  # Starting position 
		self.terminal_state = (7, 7)  # Goal position
		
		# Create an 8x8 grid with holes
		self._lake = np.zeros(self.shape, dtype=bool)  # False = frozen, True = hole
		self.hole_positions = []
		
		# Generate random hole positions
		if self.is_hardmode:
			while len(self.hole_positions) < numhole_positions:
				new_row = np.random.randint(0, 8)
				new_col = np.random.randint(0, 8)
				state = (new_row, new_col)
				
				if (
					(state not in self.hole_positions)
					and (state != self.start_state)
					and (state != self.terminal_state)
				):
					self._lake[new_row, new_col] = True  # Mark as hole
					self.hole_positions.append(state)

		# Calculate transition probabilities and rewards
		self.nS = self.shape[0] * self.shape[1]  # Number of states (64 for 8x8)
		self.nA = 4  # Four actions: UP, RIGHT, DOWN, LEFT
		
		self.P = {}
		for s in range(self.nS):
			position = np.unravel_index(s, self.shape)
			self.P[s] = {a: [] for a in range(self.nA)}
			self.P[s][0] = self._calculate_transition_prob(position, [-1, 0])  # UP
			self.P[s][1] = self._calculate_transition_prob(position, [0, 1])   # RIGHT
			self.P[s][2] = self._calculate_transition_prob(position, [1, 0])   # DOWN
			self.P[s][3] = self._calculate_transition_prob(position, [0, -1])  # LEFT

	def _calculate_transition_prob(self, position, action):
		new_row = position[0] + action[0]
		new_col = position[1] + action[1]
		new_position = (new_row, new_col)
		
		# Check for boundaries
		if 0 <= new_row < self.shape[0] and 0 <= new_col < self.shape[1]:
			if self._lake[new_row, new_col]:  # If entering a hole
				return [(1.0, new_position, -1, True)]  # Falling in hole leads to loss
			else:
				return [(1.0, new_position, 0, False)]  # Safe move
		else:
			return [(1.0, position, 0, False)]  # Invalid mov

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
					action = np.random.choice([0, 1, 3], p=[1 / 2, 1 / 4, 1 / 4])
				elif action == 1:
					action = np.random.choice([0, 1, 2], p=[1 / 4, 1 / 2, 1 / 4])
				elif action == 2:
					action = np.random.choice([1, 2, 3], p=[1 / 4, 1 / 2, 1 / 4])
				elif action == 3:
					action = np.random.choice([0, 2, 3], p=[1 / 4, 1 / 4, 1 / 2])

		return super().step(action)

	def _render_gui(self, mode):
		if self.window_surface is None:
			pygame.init()

			if mode == "human":
				pygame.display.init()
				caption = "Dead Man Drifting"
				global Tname
				if Tname:
					if Tname == '1': caption += " - Nine Thou"
					else: caption += " - " + Tname
				pygame.display.set_caption(caption)
				self.window_surface = pygame.display.set_mode(self.window_size)
			else:  # rgb_array
				self.window_surface = pygame.Surface(self.window_size)

		if self.clock is None:
			self.clock = pygame.time.Clock()

		# Load images for elements
		if self.elf_images is None:
			self.elf_images = [
				pygame.transform.scale(pygame.image.load(path.join(image_path, "img/cab_rear.png")), self.cell_size),
				pygame.transform.scale(pygame.image.load(path.join(image_path, "img/cab_right.png")),
									   self.cell_size),
				pygame.transform.scale(pygame.image.load(path.join(image_path, "img/cab_front.png")),
									   self.cell_size),
				pygame.transform.scale(pygame.image.load(path.join(image_path, "img/cab_left.png")),
									   self.cell_size),
			]

		if self.hole_img is None:
			self.hole_img = pygame.transform.scale(pygame.image.load(path.join(image_path, "img/hole.png")),
												   self.cell_size)

		if self.ice_img is None:
			self.ice_img = pygame.transform.scale(pygame.image.load(path.join(image_path, "img/ice.png")),
												  self.cell_size)

		if self.goal_img is None:
			self.goal_img = pygame.transform.scale(pygame.image.load(path.join(image_path, "img/goal.png")),
												   self.cell_size)

		# Render the FrozenLake
		for s in range(self.nS):
			row, col = np.unravel_index(s, self.shape)
			pos = (col * self.cell_size[0], row * self.cell_size[1])

			self.window_surface.blit(self.ice_img, pos)  # Draw the ice background

			if (row,col) in self.hole_positions:  # Check for holes
				self.window_surface.blit(self.hole_img, pos)

			if s == self.start_state:  # Render starting position
				self.window_surface.blit(self.start_img, pos)  # Optional starting position image
			if s == self.nS - 1:  # Render goal position
				self.window_surface.blit(self.goal_img, pos)

			if s == self.s:  # Render player (elf or froggie)
				elf_pos = (pos[0], pos[1] - 0.1 * self.cell_size[1])  # Small adjustment for positioning
				last_action = self.lastaction if self.lastaction is not None else 2  # Default to down
				self.window_surface.blit(self.elf_images[last_action], elf_pos)

		# Update display based on mode
		if mode == "human":
			pygame.event.pump()
			pygame.display.update()
			self.clock.tick(self.metadata["render_fps"])  # Control the frame rate
		else:  # rgb_array
			return np.transpose(
				np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
			)

	def DJ(self, Path):
		global start_position, Tname, pth
		pygame.mixer.init()

		def get_saved_position(pth, Len):
			if path.exists(pth):
				with open(pth) as f:
					return int(f.read().split('.')[0]) % round(int(Len))
			return 0

		pygame.mixer.music.load(Path)
		start_position = get_saved_position(pth, pygame.mixer.Sound(Path).get_length())
		pygame.mixer.music.play(start=start_position, loops=-1)
		
	def close(self):
		saveST()
		super().close()

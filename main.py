from source import FrozenLake
import numpy as np
from enum import Enum
# Create an environment
max_iter_number = 1000


class action(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


env = FrozenLake(render_mode="human", map_name="8x8")
observation, info = env.reset(seed=30)

value_list = []


def is_valid(x, y):
    return 0 <= x < 8 and 0 <= y < 8


def find_value_for_each_state():
    V = np.zeros(env.nS)
    policy = np.zeros(env.nS, dtype=int)
    iteration = 0
    gamma = .8
    V[63] = 100
    for __ in range(0, 800):
        new_V = np.copy(V)
        for state in range(0, env.nS):
            action_values = []
            for a in range(0, env.nA):
                q_value = 0
                for a2 in range(0, env.nA):
                    [_, next_pos, hole, done] = env.P[state][a2][0]
                    next_state = 8 * next_pos[0] + next_pos[1]

                    if (a == a2):
                        prob = 1 / 2
                    else:
                        prob = 1 / 4

                    reward = -1
                    if (hole == -1 and done):
                        # go in hole
                        reward = -100
                    if (next_state == 63):
                        # goal_state
                        reward = 500

                    q_value += prob * (reward + gamma * V[next_state])
                action_values.append(q_value)

            new_V[state] = max(action_values)
            policy[state] = np.argmax(action_values)
        V = new_V

    print(V.reshape((8, 8)).astype(int))
    print(policy.reshape((8, 8)))
    enum_policy = np.array([[action(policy_value).name for policy_value in row] for row in policy.reshape((8, 8))])
    for row in enum_policy:
        print(row)
    return V, policy


if __name__ == "__main__":
    V, policy = find_value_for_each_state()
    current_state = env.s
    finalreward = 0
    for __ in range(max_iter_number):
        action = policy[env.s]
        next_state, reward, done, truncated, info = env.step(action)
        finalreward -=1
        if(next_state[0]*8 + next_state[1] == 63):
            print(f"final reward {finalreward+100}")
            env.close()
            break
        if done or truncated:
            finalreward = 0
            observation, info = env.reset()

    env.close()

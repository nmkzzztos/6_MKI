import numpy as np
import random

class Gridworld:
    def __init__(self, grid_size, start, goal, obstacles):
        self.grid_size = grid_size
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.state = start
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.rewards = np.zeros(grid_size)
        self.rewards[goal] = 1
        for obs in obstacles:
            self.rewards[obs] = -1
    
    def reset(self):
        self.state = self.start
        return self.state
    
    def step(self, action):
        x, y = self.state
        if action == 'UP':
            x = max(0, x - 1)
        elif action == 'DOWN':
            x = min(self.grid_size[0] - 1, x + 1)
        elif action == 'LEFT':
            y = max(0, y - 1)
        elif action == 'RIGHT':
            y = min(self.grid_size[1] - 1, y + 1)

        if (x, y) in self.obstacles:
            reward = -1
        else:
            reward = self.rewards[x, y]
        
        self.state = (x, y)
        done = (self.state == self.goal)
        
        return self.state, reward, done
    
    def render(self):
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                if (i, j) == self.start:
                    print('S', end=' ')
                elif (i, j) == self.goal:
                    print('G', end=' ')
                elif (i, j) in self.obstacles:
                    print('X', end=' ')
                else:
                    print('.', end=' ')
            print()

def q_learning(env, episodes, alpha, gamma, epsilon):
    q_table = np.zeros((*env.grid_size, len(env.actions)))
    
    for _ in range(episodes):
        print(f'Episode {_ + 1}/{episodes}')
        state = env.reset()
        done = False
        
        while not done:
            if random.uniform(0, 1) < epsilon:
                action_idx = random.choice(range(len(env.actions)))
            else:
                action_idx = np.argmax(q_table[state])
            
            next_state, reward, done = env.step(env.actions[action_idx])
            best_next_action = np.argmax(q_table[next_state])
            td_target = reward + gamma * q_table[next_state][best_next_action]
            td_delta = td_target - q_table[state][action_idx]
            q_table[state][action_idx] += alpha * td_delta
            
            state = next_state
    
    return q_table
    

grid_size = (5, 5)
start = (0, 0)
goal = (4, 3)
obstacles = [(1, 1), (1, 3), (3, 1), (3, 3)]

env = Gridworld(grid_size, start, goal, obstacles)
env.render()
episodes = 100
alpha = 0.1
gamma = 0.9
epsilon = 0.1

q_table = q_learning(env, episodes, alpha, gamma, epsilon)

print("Trained Q-Table:")
for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        print(f"State {(i, j)}: {q_table[(i, j)]}")

env.render()

state = env.reset()
done = False
steps = 0

while not done and steps < 20:
    action_idx = np.argmax(q_table[state])
    state, reward, done = env.step(env.actions[action_idx])
    print(f"Step {steps}: State {state}, Action {env.actions[action_idx]}, Reward {reward}")
    steps += 1

if done:
    print("Reached the goal!")
else:
    print("Did not reach the goal within the step limit.")

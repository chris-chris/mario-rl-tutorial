import gym
import ppaquette_gym_super_mario
env = gym.make('ppaquette/SuperMarioBros-1-1-v0')
observation = env.reset()
for _ in range(1000):
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)

import gym

from stable_baselines import DQN
from stable_baselines.common.evaluation import evaluate_policy


# Create environment
env = gym.make('Breakout-v4')

# Instantiate the agent
model = DQN('MlpPolicy', env, learning_rate=1e-3, prioritized_replay=True, verbose=1)
# Train the agent
model.learn(total_timesteps=int(2e5))
# Save the agent
model.save(r"C:\Users\aeunal\Pictures\itü\ai\Project\models\dqn_breakout")
del model  # delete trained model to demonstrate loading

# Load the trained agent
model = DQN.load(r"C:\Users\aeunal\Pictures\itü\ai\Project\models\dqn_breakout")

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

import time
# Enjoy trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    time.sleep(0.001)
import gym
from stable_baselines import HER, SAC
from pick_and_place import FetchPickAndPlaceEnv

import time

env = gym.make("FetchPickAndPlace-v1")

goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE

# Wrap the model
model = HER('MlpPolicy', env, SAC, n_sampled_goal=4,
            goal_selection_strategy='future',
            verbose=1, buffer_size=int(1e6),
            learning_rate=0.001,
            gamma=0.95, batch_size=256,
            ent_coef='auto',
            random_exploration=0.3,
            learning_starts= 1000,
            train_freq= 1,
            policy_kwargs=dict(layers=[256, 256, 256]),
            tensorboard_log="./OpenAI/")
# Train the model
model.learn(int(8e6))

model.save("./model2")

# WARNING: you must pass an env
# or wrap your environment with HERGoalEnvWrapper to use the predict method
model = HER.load('./model2', env=env)

obs = env.reset()
episodes = 0
successes = 0
step = 0
while(episodes < 50):
    step += 1
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    env.render()
    if done or step > 1000:
        obs = env.reset()
        episodes +=1
        if _['is_success']:
            successes += 1

print('success_rate = ' + str(successes/episodes))
from pick_and_place import FetchPickAndPlaceEnv
import numpy as np

env = FetchPickAndPlaceEnv()

for counter in range(10):
    env.reset()
    for _ in range(500):
        env.render()
        env.step(env.action_space.sample())




target = np.array([[0,0,0,1], [0,0,0,1]])
for i in range(100):
    env.step(target)
    env.render()

env.step(target)

target = np.array([-0.1, -0.1, -0.1, -0.1])
for i in range(1000):
    env.step(target)
    env.render()
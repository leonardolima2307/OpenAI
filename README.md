# OpenAI
We're gonna solve a few environments from OpenAi Gym using machine learning with sklearn. 

In Acrobot_gen_1
1) I first collect the data using random steps in the environment picking up the samples with the "best" reward, in case Acrobot-v1 we pick -400 as best reward

2) The next step is clean the data, reshape it for be used by the classfifier, in this case I use SVC(kernel="linear", C=0.025) but feel free to change for another model, in the code is commented a few others models that can be tune and be implemented. 

3) The last step is run the game a couple of times and insted of using random steps, we use the previous observation and predict the action, so the model will control the robot.

The problem is solved in the first gen, depending on the reward you can run the model 100 times or 15000, I recomend not render if you're running too many times.

In the others Acrobot files I develop many generation of the model, the logic is that after you run the first model, just save the data again and pick a better reward, then train a new model and run the game using that model. this operation can be done many times, that's the meaning of reinforcement learning.





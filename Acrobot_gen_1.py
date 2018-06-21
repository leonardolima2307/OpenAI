import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
import pickle

env = gym.make("Acrobot-v1")
env.reset()
goal_steps = 600
score_requirement = -400
initial_games = 15000

def initial_population():
    training_data = []
    # all scores:
    scores = []
    accepted_scores = []
    # iterate through however many games we want:
    for game_number in range(initial_games):
        score = 0
        game_memory = []
        prev_observation = []
        for _ in range(goal_steps):
            action = random.randrange(0,3)
            observation, reward, done, info = env.step(action)
            # notice that the observation is returned FROM the action
            # so we'll store the previous observation here, pairing
            # the prev observation to the action we'll take.
            if len(prev_observation) > 0 :
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score += reward
            if done: 
                #print('score is', score)
                break
        if score >= score_requirement:
            print('score is', score, 'Game number', game_number)
            accepted_scores.append(score)
            for data in game_memory:  
                # saving our training data
                ##print our data
               # print('data is', data)
                training_data.append([data[0], data[1]])
        # reset env to play again
        env.reset()
        # save overall scores
        scores.append(score)

    ##saving training data
    filename = r"C:\Users\lima.l\Documents\python\Deep Learning\q learning\training_data_acrorobot.sav"
    pickle.dump(training_data, open(filename, 'wb'))
    print('training_data saved Saved')
    return training_data

training_data = initial_population()
#Transforming training_data to be used by the classifier
features = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
labels = [i[1] for i in training_data]
print('features', features.shape)
nsamples, nx, ny = features.shape
features2d = features.reshape((nsamples,nx*ny))
### TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    features2d, labels, test_size=0.1, random_state=42)
print(len(X_train))
##training
print('Start Training :)')
from sklearn.svm import SVC
#clf = SVC(kernel="linear", C=0.025)
#clf = AdaBoostClassifier()
clf = KNeighborsClassifier(3)
#print('2d', features2d[0])
#clf.fit(features2d, labels) 
clf.fit(X_train, y_train)
print('Finish Training')
score = clf.score(X_test, y_test)
print('score of the model is', score)

##PLAYING
scores = []
for each_game in range(10):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()
    for _ in range(goal_steps):
        env.render()

        if len(prev_obs)==0:
            action = random.randrange(0,2)
        else:
            #predicting 
            predict = np.array(prev_obs)
            #print('predict playing', ''.join(map(str, clf.predict(predict.reshape(1, -1)))))
            action = clf.predict(predict.reshape(1, -1))[0]   
        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        game_memory.append([new_observation, action])
        score+=reward
        if done: break

    scores.append(score)

print('Average Score:',sum(scores)/len(scores))
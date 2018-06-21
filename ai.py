#We are only making the AI to select the right action at the right time
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

class Network(nn.Module):
    #TODO: PLay with the architecture of the neural network. 
    #Perhaps make this configureable by making inputs to this class
    
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__() 
        #this variable is the number of input neurons
        self.input_size = input_size
        #variable for the number of output neurons
        self.nb_action = nb_action
        #full connections between layers. All th neurons of the hidden layer
        #be connected to all the connnections of the input layer
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
         



    #the function that perfroms the forward propogation
    def forward(self, state):
        #hidden neurons
        #We use the rectofier activation function, to activate the hidden layer
        x = F.relu(self.fc1(state))
        
        #return the output neurons, which are the actions (the q_values more specifically)
        q_values = self.fc2(x)
        return q_values


#This class is for implementing experience replay
class ReplayMemory(object):
    
    #constructor
    def __init__(self, capacity):
        
        #the maximum number of transitions we have saved of events
        self.capacity = capacity
        #contain the last 'n' number of events
        self.memory = []
    #append a new event/transition in the memory
    def push(self, event):
        self.memory.append(event)
        
        #make sure we dont exceed the number of saved transitions
        if len(self.memory) > self.capacity:
            del self.memory[0]
            
            
    #return a randon sample from our memory 
    def sample(self, batch_size):
        #taking random samples of the memory that are of size 'batch_size'
        #if list = ((1, 2, 3)(4, 5, 6)) -> zip(*list) = ((1, 4), (2, 3), (5,6))
        #our data format for this problem is ((states), (actions), (rewards))
        samples = zip(*random.sample(self.memory, batch_size))
        #We have to concatinate the samples above with regards to the first dimension, states.
        #We do this for alignment, so that we are getting a list of batches such that each
        #state, action, and reward batch corrasponds to the same time 't', with each formatted 
        #into a PyTorch variable.
        return map(lambda x: Variable(torch.cat(x, 0)), samples)
    

#now we implement the deep Q-Learning model
class Dqn():
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        
        #will be be a revolving window of the last 'n' rewards, as a running average.
        self.reward_window = []
        
        #the neural network
        self.model = Network(input_size, nb_action)
        
        #create the memory for experiance replay
        self.memory = ReplayMemory(100000)
        
        #optimizer provided by torch to perform stochastic gradient decent backpropogation
        #TODO: Make adjustable learning rate
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        
        #Creates the batch structure of states. Essentially, the NN requires we place into it a batch of states. 
        #So the data structure will resemble a tuple: (0, (state dimentions....))
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        
        #actions are either 0, 1, or 2. These map to the indexes of the rotation-angle array
        self.last_action = 0
        
        self.last_reward = 0
        
    #this is where we utilize the output of our neural network to generate an action
    def select_action(self, state):
        #Here we implement a softmax propability distribution instead of Argmax as in the typical inplementation of the Q-Learning Algorithm 
        #We assign a high probability to  the highest Q-Value action available, but give a tempurature for other 
        #action's which will decrease over time. THe higher the tempurature at the start will allow for randomly 
        #selected action's to actually be utilized. Over time, the temperature will decrease based of a tempurature  equation
        
        #TODO: Variable class has been depreciated. Reimplement with Tensor constructor.
        #TODO: Allow for configuration of Softmax tempurature param
        probs = F.softmax(self.model(Variable(state, volatile = True)) * 7) 
        
        #random selction from our probability distribution
        action = probs.multinomial()
        
        return action.data[0, 0]
    
    #Here is where the training of the DQN will actually occur. Forward and backpropogation with stochastic gradient descent
    #will be inplemented to determin the relative affect of all our weights for each node. 
    #Q(s_t, a_t) == Q(s_t, a_t) + alpha[r_(t+1) + gamma * {MAX_a Q(s_(t+1), a)} - Q(s_t, a_t)]
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        
        #Our network is expecting a batch of states. Then we gather together all the sected best actions from the NN
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(0)).squeeze(1)
        
        #We take the maximum of the next state's Q-Vsalues with respect to the next states actions,
        #where actions are stored in index 1 and states stored in index 0
        next_outputs = self.model(batch_next_state).detact().max(1)[0]
        
        #r_(t+1) + gamma * {MAX_a Q(s_(t+1), a)} - Q(s_t, a_t)
        #Target in one sample in memory
        target = self.gamma*next_outputs + batch_reward
        
        #Huber loss error function
        td_loss = F.smooth_l1_loss(outputs, target)
        
        #here we will take the loss to perfrom back propogation on our weights with the optimizaer functions
        self.optimizer.zero_grad();
        
        #backpropogate
        td_loss.backward(retain_variables = True)
        
        #update weights based on contribution to error. long Derivatives made easy.
        self.optimizer.step()
    
    
    def update(self, reward, new_signal): 
        
        #We got to convert our simple array of environment variables, and then turn into a torch tensor
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        
        #We have a brand new transition, so update the memory
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        
        #We need to play an action
        action = self.select_action(new_state)
        
        #Allow the NN to learn from random samples of transitions
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_reward, batch_action = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
            
        
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        
        #update the reward window, to show the evolving mean of rewards
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            #delete the oldest entry
            del self.reward_window[0]
        
        return action
    
    #mean of the reward in the sliding window
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window) + 1.)
    
    #Saves the NN
    def save(self):
        
        #we will save our model and the optimizer to a file named 'last_brain.pth'
        torch.save({ 'state_dict': self.model.state_dict(), 'optimizer' : self.optimizer.state_dict }, 'last_brain.pth' )
    
    #load saved NN
    def load(self):
        
        if os.path.isfile('last_brain.pth'):
            print('=> loading checkpoint...')
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Done!")
        else:
            print('No model and optimizer found...')
            
        
        
        
    
        
        


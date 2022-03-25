import random, sys, getopt
from collections import deque, namedtuple
from datetime import datetime
import gym
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim


class VariableDQN(nn.Module):
    def __init__(self, obs_dims: list, action_dims: list, hWidth: int, hDepth: int, lr: float):
        """Creates a variable dimension DQN

        Args:
            obs_dims (list): dimensionality of observation (feature) space
            action_dims (list): dimensionality of action (output) space
            hWidth (int): # of nodes in each hidden layer
            hDepth (int): # of hidden layers
            lr (float): learning rate
        """
        super(VariableDQN, self).__init__()
        layers = [nn.Linear(*obs_dims, hWidth),nn.ReLU()]
        for l in range(hDepth):
            layers += [nn.Linear(hWidth, hWidth),nn.ReLU()]
        layers += [nn.Linear(hWidth, action_dims)]
        
        self.net = nn.Sequential(*layers)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(device=self.device)

    def forward(self, x):
        return self.net(x.float())

SRM = namedtuple('SRM', ('state0', 'state1', 'action', 'reward', 'done'))

class DQNAgent():
    def __init__(self, env_name: str, DQN_dim: tuple, mem_cap: int = 10000, batch_size: int = 32):
        """Creates a specific instance of a VariableDQN

        Args:
            env_name (str): name of the OpenAI Gym game the DQN Agent will learn
            DQN_dim (tuple): the dimensionality of the DQN's hidden layers - (hWidth, hDepth)
            mem_cap (int, optional): max # of most recent SRM stored. Defaults to 10000.
            batch_size (int, optional): # of SRM trained on each step. Defaults to 32.
        """
        # the corresponding OpenAI Gym env must be installed
        self.env = gym.make(env_name)
        
        # inherits the observation/action space from the game it's playing
        self.obs_dim = self.env.observation_space.shape if len(self.env.observation_space.shape) else [self.env.observation_space.n]
        self.action_dim = self.env.action_space.n
        (hWidth, hDepth) = DQN_dim
        print(f'game: {env_name}\nobs_dim: {self.obs_dim}\naction_dim: {self.action_dim}\nDQN_dim: {DQN_dim}\n')
        
        self.gamma = 0.99
        self.epsilon = 1
        self.eps_min = 0.01
        self.lr = 0.001

        self.mem_cap = mem_cap
        self.batch_size = batch_size

        self.total_steps = 0
        self.decay = 0.99


        # deep q network
        self.Q = VariableDQN(obs_dims=self.obs_dim, action_dims=self.action_dim, hWidth=hWidth, hDepth=hDepth, lr=self.lr)

        # state reward memory
        self.srm = deque([], maxlen=self.mem_cap)

    def save_memory(self, state0: list, state1: list, action: int, reward: float, done: bool):
        self.srm.append(SRM(state0, state1, action, float(reward), done))


    def get_action(self, state: list) -> int:
        """Selects the agent's action according to an epsilon greedy regime
        (will return either a random action or the action predicted by the current 
        state of the DQN to produce the greatest reward)
        Args:
            state (list): current observed state of the agent's environment

        Returns:
            int: action #
        """
        if (random.random() < self.epsilon):
            return self.env.action_space.sample()
        else:
            return T.argmax(self.Q.forward(T.tensor(state).to(device=self.Q.device))).item()

    def train_once(self):
        if len(self.srm) < self.batch_size: return 1
        self.epsilon = max(self.eps_min, self.decay**self.total_steps)
        
        # get batch of prior state-reward memories to train on
        batch = SRM(*zip(*random.sample(self.srm, self.batch_size)))
        batchi = np.arange(self.batch_size, dtype=np.int32)

        self.Q.optimizer.zero_grad()
        q_state0 = self.Q.forward(T.tensor(np.array(batch.state0)).to(device=self.Q.device))[batchi,batch.action]
        q_state1 = self.Q.forward(T.tensor(np.array(batch.state1)).to(device=self.Q.device))
        q_state1[T.tensor(batch.done).to(device=self.Q.device)] = 0

        reward = T.tensor(batch.reward).to(device=self.Q.device)
        q_target = reward + self.gamma*T.max(q_state1, dim=1)[0]
        loss = self.Q.loss(q_target, q_state0).to(device=self.Q.device).float()
        ret = loss.item()
        loss.backward()
        self.Q.optimizer.step()
        return ret
        
    def train(self, epochs: int, render: bool = False) -> tuple:
        """Trains the DQNAgent

        Args:
            epochs (int): # of epochs in training session
            render (bool, optional): if True, will render every 100th attempt. Defaults to False.

        Returns:
            tuple: (list(reward_history), list(epsilon_history), list(loss_history))
        """
        reward_history = []
        epsilon_history = []
        loss_history = []
        
        # run a full simulation for each epoch
        while self.total_steps < epochs:
            currentState = self.env.reset()
            done = False
            totalReward = 0
            loss = 0
            while not done:
                if render and not self.total_steps % 100: self.env.render()
                
                # make an action and store the state reward memory
                action = self.get_action(currentState)
                nextState, reward, done, _ = self.env.step(action)
                totalReward += reward
                self.save_memory(currentState, nextState, action, reward, done)
                currentState = nextState
                
                # train on prior memories
                loss = self.train_once()
                
            reward_history.append(totalReward)
            epsilon_history.append(self.epsilon)
            loss_history.append(loss)
            if not self.total_steps % 100:
                print(
                    f'ts: {datetime.now()} | epoch: {self.total_steps} | last rew: {reward_history[-1]} | avg rew: {sum(reward_history[-30:])/len(reward_history[-30:])} | eps: {self.epsilon}') 
            self.total_steps += 1
            
        return reward_history, epsilon_history, loss_history

def run_experiment(output: str, widths: list, depths: list, epochs: int, render: bool):
    """Run a complete experiment on multiple variable size DQNs across multiple Gym games

    Args:
        output (str): output filename (written in ./outputs/)
        widths (list): list of different hidden layer widths to be tested
        depths (list): list of different hidden layer depths to be tested
        epochs (int): # of epochs to train for
        render (bool): render every 100th training epoch
    """
    games = ['CartPole-v1','LunarLander-v2','Acrobot-v1']
    print(f'running experiment with\n\twidths: {widths}\n\tdepths: {depths}\n\tepochs: {epochs}')
    print(f'writing outputs to "./outputs/{output}.txt"')
    print(f'rendering is {render}\n')
    
    for game in games:
        for w in widths:
            for d in depths:
                t = DQNAgent(game, DQN_dim=(w,d))
                rh, eh, lh = t.train(epochs, render)
                with open(f'./outputs/{output}.txt', "a") as f:
                    f.write(f'{game},{w},{d}:'+str([rh, eh, lh])+'\n')

if __name__ == '__main__':
    # TODO: allow widths/depths as args
    try:
        opts, args = getopt.getopt(sys.argv[1:],"ho:e:r")
    except getopt.GetoptError:
        print('python3 variable_dqn.py -o <output> -e <epochs> -r [render training]')
    
    widths = [16,32,64]
    depths = [1,2,4]
    epochs = 500
    render = False
    output = 'test-output'
    
    for opt, arg in opts:
        if opt == '-h':
            print('python3 variable_dqn.py -o <output> -e <epochs> -r')
            exit(1)
        elif opt == '-o':
            output = arg
        elif opt == '-e':
            epochs = int(arg)
        elif opt == '-r':
            render = True

    run_experiment(output, widths, depths, epochs, render)
    
# Variable dimension DQN performance 
I was curious about the performance of dfferent sizes of Deep Q Networks when applied to a handful of OpenAI Gym reinforcement learning games, so I put together a simple project to test a variety of differently sized DQNs on a couple of Gym games. 
If you want to read more about my observations, you can see a short summary I wrote [here](https://drive.google.com/file/d/1cgKxDrovG6gWUBYMt3pmxFIv8A4QeZmq/view?usp=sharing).

# Setup and Usage
To install the dependencies, run

``` python3 pip -m install -r requirements.txt```

The code for creating and training the DQNs can be found in `variable_dqn.py`. You can run the experiment with default settings via

```python3 variable_dqn.py -o <output> -e <epochs> -r```

and it will start writing outputs into the output folder (see `example_output.txt`). 

After the experiment is finished, you can run 

```python3 visualize.py -o <output>```

to view the data
import sys, getopt
import matplotlib.pyplot as plt
# import numpy as np

def plotsingle(output):
    title, rewards, _,_ = output
    
    plt.xlabel('episodes')
    plt.ylabel('rewards')
    plt.title(title)
    plt.plot(rewards, 'o')
    plt.show()
    
if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:],"ho:")
    except getopt.GetoptError:
        print('python3 visualize.py -o <output>')
        
    output = f'example-output'
    
    for opt, arg in opts:
        if opt == '-h':
            print('python3 visualize.py -o <output>')
            exit(1)
        elif opt == '-o':
            output = arg
    
    results = open(f'./outputs/{output}.txt','r').readlines()
    outputs = []
    for result in results:
        title, data = result.split(':[[')
        rewards, epsilons, losses = map(lambda x: 
            list(map(float, x.split(', '))), 
            data.replace(']]','').replace('1000000000, ','').split('], ['))
        outputs.append([title, rewards, epsilons, losses])
        
    for o in outputs:
        plotsingle(o)
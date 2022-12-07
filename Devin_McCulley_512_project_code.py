import numpy as np
import pandas as pd

##### problem setup #####
# load in from csv using pandas
N = 5
initial_budget = 0
start = 0
initial_chance = (1,1,0)
locations = ['A','B','C']
electoral_votes = [1,2,3]

TSP = np.array([[0,1,2],[1,0,1],[2,1,0]])

# probability matrix
                    #lose win
P_visit = np.array([ [0.5,0.5], # lose
                     [0,1]])# win

P_no_visit = np.array([[1,0],
                       [0.1,0.9]])

##### policy evaluation: backwards algorithm #####
states = []
for next_state in range(len(locations)):
    for budget in range(initial_budget+1):
        for chance_num in range(2**len(locations)):
            outcome = (int(chance_num/4)%2, int(chance_num/2)%2, int(chance_num/1)%2)
            state = (next_state,budget,outcome)
            states.append(state)

##### policy optimization: backwards algorithm #####
# step 1
t = N
s_t = states
v = {}  # dictionary value function
p = {}  # dictionary optimal policy
for node in s_t:
    v[node] = sum([electoral_votes[i]*node[2][i] for i in range(len(node[2]))])

    location = locations[node[0]]
    budget = node[1]
    outcome = node[2]
    print(location,budget,outcome,'value',v[node])

# step 2
# TODO add comments in each for loop layer
while t!=0:
    # step 3
    #subsitute t-1 for t
    t = t-1
    v_copy = v.copy()

    print('t',t)
    
    for node in s_t:
        location = node[0]
        budget = node[1]
        outcome = node[2]

        debug = False
        # if location==2 and budget==2 and outcome==(0,0,0):
        #     debug = True

        if(debug): print(locations[location],budget,outcome)

        max_cost = -np.Inf
        max_action = None
        for action in range(len(locations)):
            cost = 0
            remaining_budget = 0# budget-TSP[location][action]
            if remaining_budget<0:      # limit actions to be within budget
                continue

            if(debug): print('action',action)

            for chance_num in range(2**len(locations)):
                next_outcome = (int(chance_num/4)%2, int(chance_num/2)%2, int(chance_num/1)%2)
                prob_visit = 1
                for next_location in range(len(locations)):
                    if next_location==action:
                        prob_visit *= P_visit[outcome[next_location],next_outcome[next_location]]
                    else:
                        prob_visit *= P_no_visit[outcome[next_location],next_outcome[next_location]]

                cost += prob_visit * v[(action,remaining_budget,next_outcome)]
                if(debug): print('action',action,'remaining_budget',remaining_budget,'next_outcome',next_outcome,'prob_visit',prob_visit,'v',v[(action,remaining_budget,next_outcome)],'cost',cost)
                
            if(debug): print('summed cost',cost)

            if cost > max_cost:
                max_cost = cost
                max_action = action

        v_copy[node] = max_cost
        p[node] = max_action
    
    v = v_copy

print("\nProblem 3")
print("Optimal expected value:",v[(start,initial_budget,initial_chance)])
print("Optimal policy:\nState | Action\n--------------")
for key,value in p.items():
    location = locations[key[0]]
    budget = key[1]
    outcome = key[2]
    print(location,budget,outcome,'= action:',locations[value],'value',v[key])

# TODO backtrack

# list of states to go to

# action travel to C
# A 2 (1, 1, 0)  ->   C 0 (1, 1, 1)
# A 2 (1, 1, 0)  ->   C 0 (1, 1, 0)
# A 2 (1, 1, 0)  ->   C 0 (0, 1, 0)
# A 2 (1, 1, 0)  ->   C 0 (0, 0, 0) 

# Decision Tree
xs = []
ys = []
for key,value in p.items():
    location = key[0]
    budget = key[1]
    outcome = key[2]
    action = value
    x = [location,budget,outcome[0],outcome[1],outcome[2],outcome[location]]
    y = [action]

    xs.append(x)
    ys.append(y)

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
clf = DecisionTreeClassifier(max_depth = None).fit(xs, ys)
plot_tree(clf, filled=True,impurity=False, class_names=locations, feature_names = ["current location","budget","outcome_A","outcome_B","outcome_C","outcome_current"])

plt.title("Decision tree")
plt.show()
fig.savefig('decision_tree.png')


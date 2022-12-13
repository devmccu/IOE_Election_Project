import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# our project:
# run finite MDP
# report policy
#   decision tree
#   table
# compare finite MDP optimized policy to random policy
#   simulation: random start, budget, and outcome
#   transtion randomly and record the reward
# visualize
#   Support vector machine to show difference in our model
#   plot showing area gained or loss on expected vs actual final votes

# option 1:
# infinite MDP 
# report policy (decision tree)

# option 2:
# finite MDP using hidden transformation matrix  (assume always win)
# report the travl path

# option 3:
# made a decision on when to start traveling 
# find the N value where action != current_location     # TODO iterate over values of N

# example path (either from simulation or outcome of finite MDP)
# t=1 A, 2, 1,1,0 a=C -> 
# t=2 C, 0, 1,1,1 a=C ->   
# t=3 C, 0, 1,1,1 ====== final score 3
# A -> C -> C
# record reward and average across all trials


##### problem setup #####
# TODO load in from csv using pandas

# N = 2
# use_budget = False
# initial_budget = 2
# start = 0
# initial_chance = (1,1,0)
# locations = ['A','B','C']
# electoral_votes = [1,2,3]
# TSP = np.array([[0,1,2],[1,0,1],[2,1,0]])

N = 3
use_budget = True
initial_budget = 9
start = 2
initial_chance = (1,0,1,0,1)
locations = ['IL','IN','MI','OH','WI']
electoral_votes = [20,11,16,18,10]
TSP = np.array([[0,3,6,7,5],
                [3,0,5,4,7],
                [6,5,0,4,6],
                [7,4,4,0,9],
                [5,7,6,9,0]])

# probability matrix
                   # lose win
P_visit = np.array([ [0.5,0.5], # lose
                     [0,  1]])  # win

                      # lose win
P_no_visit = np.array([ [1,  0],    # lose
                        [0.1,0.9]]) # win

##### states setup #####
states = []
if(use_budget):
    for next_state in range(len(locations)):
        for budget in range(initial_budget+1):
            for chance_num in range(2**len(locations)):
                outcome = tuple([int(chance_num/2**digit)%2 for digit in range(len(locations)-1,-1,-1)])
                state = (next_state,budget,outcome)
                states.append(state)
else:
    for chance_num in range(2**len(locations)):
        outcome = tuple([int(chance_num/2**digit)%2 for digit in range(len(locations)-1,-1,-1)])
        state = (start,initial_budget,outcome)
        states.append(state)

print(len(states),"states")

##### policy optimization: backwards algorithm #####
# step 1
t = N
s_t = states
v = {}  # dictionary value function
p = {}  # dictionary optimal policy
for node in s_t:
    location = locations[node[0]]
    budget = node[1]
    outcome = node[2]

    v[node] = sum([electoral_votes[i]*node[2][i] for i in range(len(node[2]))])
    p[node] = node[0]
    
    # print(location,budget,outcome,'value',v[node])

# step 2
# TODO add comments in each for loop layer
while t!=0:
    # step 3: subsitute t-1 for t
    t = t-1
    v_copy = v.copy()
    
    for node in s_t:
        location = node[0]
        budget = node[1]
        outcome = node[2]

        debug = False

        if(debug): print(locations[location],budget,outcome)

        max_cost = -np.Inf
        max_action = None
        for action in range(len(locations)):
            cost = 0
            remaining_budget = 0
            if(use_budget):
                remaining_budget = budget-TSP[location][action]
                if remaining_budget<0:      # limit actions to be within budget
                    continue
            
            if(debug): print('action',action)

            for chance_num in range(2**len(locations)):
                next_outcome = tuple([int(chance_num/2**digit)%2 for digit in range(len(locations)-1,-1,-1)])
                prob_visit = 1
                for next_location in range(len(locations)):
                    if next_location==action:
                        prob_visit *= P_visit[outcome[next_location],next_outcome[next_location]]
                    else:
                        prob_visit *= P_no_visit[outcome[next_location],next_outcome[next_location]]

                if use_budget:
                    cost += prob_visit * v[(action,remaining_budget,next_outcome)]
                else:
                    cost += prob_visit * v[(start,initial_budget,next_outcome)]
                    
                if(debug): print('action',action,'remaining_budget',remaining_budget,'next_outcome',next_outcome,'prob_visit',prob_visit,'v',v[(action,remaining_budget,next_outcome)],'cost',cost)
                
            if(debug): print('summed cost',cost)

            if cost > max_cost:
                max_cost = cost
                max_action = action

        v_copy[node] = max_cost
        p[node] = max_action
    
    v = v_copy


##### print results #####
start_node = (start,initial_budget,initial_chance)
print("Optimal expected value:",round(v[start_node],3),'action:',locations[p[start_node]])
if False:
    print("Optimal policy:")
    for key,value in p.items():
        location = locations[key[0]]
        budget = key[1]
        outcome = key[2]
        print(location,budget,outcome,'= action:',locations[value],'value',v[key])


##### policy evaluation: random #####
def evaluate_random(start,initial_budget,initial_chance):
    trial_num = 1000
    sum_reward = 0
    for ii in range(trial_num):  # sample for each trial number
        node = (start,initial_budget,initial_chance)
        for t in range(N):      # run through finite MDP
            location = node[0]
            budget = node[1]
            outcome = node[2]

            valid_actions = TSP[location]<=budget   # choose a random valid action
            actions_prob = valid_actions/valid_actions.sum()
            action = np.random.choice(range(len(locations)), p=actions_prob)

            # apply transition probabilities
            next_outcome = []
            for next_location in range(len(locations)):
                if next_location==action:
                    next_location_outcome = np.random.choice([0,1],p=P_visit[outcome[next_location]])
                else:
                    next_location_outcome = np.random.choice([0,1],p=P_no_visit[outcome[next_location]])
                next_outcome.append(next_location_outcome)

            node = (action,budget-TSP[location][action],tuple(next_outcome))

        sum_reward += sum([electoral_votes[i]*node[2][i] for i in range(len(node[2]))])
    return round(sum_reward/trial_num,3)

random = evaluate_random(start,initial_budget,initial_chance)
print("Random policy expected value:",random)


# policy evaluation for each starting condition
for node in s_t:
    location = node[0]
    budget = node[1]
    outcome = node[2]

    MDP = v[node]
    random = evaluate_random(location,budget,outcome)

    print(locations[location],budget,outcome,'= MDP:',MDP,'random:',random)

    plt.scatter(MDP,random)

# plot random vs MDP on plot
plt.show()

################# Decision Tree #################
if True:
    xs = []
    ys = []
    feature_names = []
    for key,value in p.items():
        location = key[0]
        budget = key[1]
        outcome = key[2]
        action = value
        x = [location,budget,outcome[location]]    # highest losing electoral state
        feature_names = ["current location","budget","outcome_current"]
        for idx_outcome in range(len(outcome)):
            x.append(outcome[idx_outcome])
            feature_names.append("outcome "+locations[idx_outcome])
        y = action

        xs.append(x)
        ys.append(y)


    # locations = ["left","stay","right"]

    import matplotlib.pyplot as plt
    from sklearn.tree import plot_tree
    from sklearn.tree import DecisionTreeClassifier

    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
    clf = DecisionTreeClassifier(max_depth = None).fit(xs, ys)
    plot_tree(clf, filled=True,impurity=False, class_names=locations, feature_names = feature_names)

    plt.title("Decision tree")
    plt.show()
    fig.savefig('decision_tree.png')


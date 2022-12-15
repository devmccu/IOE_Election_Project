import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

################# Overview #################
"""
run finite MDP
given N, locations, travel_costs, budget, EC votes, and party
calculate optimal policy with backwards induction
  print policy
  show decision tree
  # TODO save to csv
compare finite MDP optimized policy to random policy
  simulate many trials: random start, budget, and outcome
  use same transtion probabilies and record the average reward
  visualize MDV vs. Random
    plot seperated by start location, budget, or outcome
"""

print_policy = True
show_plot = True
show_tree = False

################# problem setup #################

# N = 2
# use_budget = True
# initial_budget = 2
# start = 0
# initial_chance = (1,1,0)
# locations = ['A','B','C']
# party = ['purple','purple','purple']
# electoral_votes = [1,2,3]
# travel_costs = np.array([[0,1,2],[1,0,1],[2,1,0]])

N = 3  # or 5
use_budget = True
initial_budget = 9
start = 2
initial_chance = (1,0,1,0,1)
locations = ['IL','IN','MI','OH','WI']
party = ['blue','red','purple','purple','purple']
electoral_votes = [20,11,16,18,10]
travel_costs = np.array([[0,3,6,7,5],
                         [3,0,5,4,7],
                         [6,5,0,4,6],
                         [7,4,4,0,9],
                         [5,7,6,9,0]])

# probability matrix
                            # lose win
P_visit_purple = np.array([ [0.6,0.4],  # lose
                            [0.3,0.7]]) # win

                               # lose win
P_no_visit_purple = np.array([ [0.8,0.2],  # lose
                               [0.4,0.6]]) # win

                         # lose win
P_visit_red = np.array([ [0.7,0.3],  # lose
                         [0.4,0.6]]) # win

                            # lose win
P_no_visit_red = np.array([ [0.9,0.1],  # lose
                            [0.5,0.5]]) # win

                          # lose win
P_visit_blue = np.array([ [0.5,0.5],  # lose
                          [0.2,0.8]]) # win

                             # lose win
P_no_visit_blue = np.array([ [0.7,0.3],  # lose
                             [0.3,0.7]]) # win

################# states setup #################
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


################# policy optimization: backwards algorithm #################
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
while t!=0:
    # step 3: subsitute t-1 for t
    t = t-1

    v_copy = v.copy()   # copy v to heep v_t unchanged until all iterations are done
    
    for node in s_t:    # for each possible state update the value function
        location = node[0]
        budget = node[1]
        outcome = node[2]

        debug = False   # optional verbose print outs
        if(debug): print(locations[location],budget,outcome)

        # variables to track the best action at this state
        max_reward = -np.Inf
        max_action = None
        for action in range(len(locations)):    # consider all actions
            remaining_budget = 0
            if(use_budget):
                remaining_budget = budget-travel_costs[location][action]
                if remaining_budget<0:      # limit actions to be within budget
                    continue
            
            if(debug): print('action',action)

            reward = 0    # calculate the reward as sum of probability times value
            for chance_num in range(2**len(locations)):     # probability is a product of visit and no visit transition matrices
                next_outcome = tuple([int(chance_num/2**digit)%2 for digit in range(len(locations)-1,-1,-1)])
                prob_visit = 1  # variable to get the overall probability of ending up with next_outcome
                for next_location in range(len(locations)):  # multiply prob_visit by all states' probability
                    # probability changes based on state classification
                    if party[next_location]=='blue':
                        P_visit = P_visit_blue
                        P_no_visit = P_no_visit_blue
                    elif party[next_location]=='red':
                        P_visit = P_visit_red
                        P_no_visit = P_no_visit_red
                    elif party[next_location]=='purple':
                        P_visit = P_visit_purple
                        P_no_visit = P_no_visit_purple

                    # one state is visited
                    if next_location==action:
                        prob_visit *= P_visit[outcome[next_location],next_outcome[next_location]]
                    # the rest are not visited
                    else:
                        prob_visit *= P_no_visit[outcome[next_location],next_outcome[next_location]]

                # accumulate reward
                if use_budget:
                    reward += prob_visit * v[(action,remaining_budget,next_outcome)]
                else:
                    reward += prob_visit * v[(start,initial_budget,next_outcome)]
                    
                if(debug): print('action',action,'remaining_budget',remaining_budget,'next_outcome',next_outcome,'prob_visit',prob_visit,'v',v[(action,remaining_budget,next_outcome)],'reward',reward)
            
            if(debug): print('summed reward',reward)

            # remember the highest reward and best action
            if reward > max_reward:
                max_reward = reward
                max_action = action

        # store the highest reward action into the policy
        v_copy[node] = max_reward
        p[node] = max_action
    
    v = v_copy  # discard v_t and use v_{t+1} going forward


################# print results #################
start_node = (start,initial_budget,initial_chance)
print("Optimal expected value:",round(v[start_node],3),'action:',locations[p[start_node]])
if print_policy:
    print("Optimal policy:")
    for key,value in p.items():
        location = locations[key[0]]
        budget = key[1]
        outcome = key[2]
        print(location,budget,outcome,'= action:',locations[value],'value',v[key])


################# policy evaluation: random #################
def evaluate_random(start,initial_budget,initial_chance):
    trial_num = 1000
    sum_reward = 0
    for ii in range(trial_num):  # sample for each trial number
        node = (start,initial_budget,initial_chance)
        for t in range(N):      # run through finite MDP
            location = node[0]
            budget = node[1]
            outcome = node[2]

            valid_actions = travel_costs[location]<=budget   # choose a random valid action
            actions_prob = valid_actions/valid_actions.sum()
            action = np.random.choice(range(len(locations)), p=actions_prob)

            # apply transition probabilities
            next_outcome = []
            for next_location in range(len(locations)):
                # probability changes based on state classification
                if party[next_location]=='blue':
                    P_visit = P_visit_blue
                    P_no_visit = P_no_visit_blue
                elif party[next_location]=='red':
                    P_visit = P_visit_red
                    P_no_visit = P_no_visit_red
                elif party[next_location]=='purple':
                    P_visit = P_visit_purple
                    P_no_visit = P_no_visit_purple

                if next_location==action:
                    next_location_outcome = np.random.choice([0,1],p=P_visit[outcome[next_location]])
                else:
                    next_location_outcome = np.random.choice([0,1],p=P_no_visit[outcome[next_location]])
                next_outcome.append(next_location_outcome)

            node = (action,budget-travel_costs[location][action],tuple(next_outcome))

        sum_reward += sum([electoral_votes[i]*node[2][i] for i in range(len(node[2]))])
    return sum_reward/trial_num

random = evaluate_random(start,initial_budget,initial_chance)
print("Random policy expected value:",round(random,3))


################# policy comparison for each starting condition #################
import matplotlib.cm as cm
if show_plot:
    random_value = {}

    for node in s_t:
        location = node[0]
        budget = node[1]
        outcome = node[2]

        rand = evaluate_random(location,budget,outcome)
        random_value[node] = rand
        
        print(locations[location],budget,outcome,'=> MDP:',round(v[node],3),'random:',round(rand,3))


    legends = []
    # sort by budget or location
    legends.append( ['budget '+str(x) for x in range(initial_budget+1)] )
    legends.append( ['start '+str(x) for x in locations] )
    legends.append( ['outcome '+tup_list for tup_list in [','.join(tuple([str(int(x/2**digit)%2) for digit in range(len(locations)-1,-1,-1)])) for x in range(2**len(locations))]] )

    for legend in legends:
        plt.figure()
        colors = cm.rainbow(np.linspace(0, 1, len(legend)))
        MDP_values = []
        random_values = []
        for idx in range(len(legend)):
            MDP_values.append([])
            random_values.append([])
        for node in s_t:
            location = node[0]
            budget = node[1]
            outcome = node[2]

            MDP_value = v[node]
            rand_value = random_value[node]

            if(legend[0][:5] == "start"):
                MDP_values[location].append(MDP_value)
                random_values[location].append(rand_value)
            elif(legend[0][:6] == "budget"):
                MDP_values[budget].append(MDP_value)
                random_values[budget].append(rand_value)
            elif(legend[0][:7] == "outcome"):
                outcome_count = sum([outcome[digit]*2**(len(locations)-1-digit) for digit in range(len(locations))])
                MDP_values[outcome_count].append(MDP_value)
                random_values[outcome_count].append(rand_value)
            else:
                print('error')

        for idx in range(len(legend)):
            plt.scatter(MDP_values[idx],random_values[idx],color=colors[idx])

        # plot random vs MDP on plot
        plt.xlabel("MDP exp. value")
        plt.ylabel("Random exp. value")
        plt.title("Expected Value for each starting condition")
        plt.legend(legend)
    plt.show()


################# Decision Tree #################
if show_tree:
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

    import matplotlib.pyplot as plt
    from sklearn.tree import plot_tree
    from sklearn.tree import DecisionTreeClassifier

    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (8,8), dpi=400)
    clf = DecisionTreeClassifier(max_depth = None).fit(xs, ys)
    plot_tree(clf, filled=True,impurity=False, class_names=locations, feature_names = feature_names)

    plt.title("Decision tree")
    plt.show()
    # fig.savefig('decision_tree.png')





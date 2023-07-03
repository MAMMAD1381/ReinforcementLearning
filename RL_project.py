import gym
import time
import numpy as np
import matplotlib.pyplot as plt
import copy
import random
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches


############################### env method : you don't need to know them


def modify_rewards(next_state, custom_map, hole_reward, goal_reward, move_reward):
    custom_map_flaten = get_flaten_custom_map(custom_map)
    state_type = custom_map_flaten[next_state]

    if state_type == "H":
        return hole_reward  # Decrease the reward for falling into a hole
    elif state_type == "G":
        return goal_reward  # Increase the reward for reaching the goal
    else:
        return move_reward  # Decrease the reward for moving


def modify_rewards_P(envP, custom_map, hole_reward, goal_reward, move_reward):
    custom_map_flaten = []
    for row in custom_map:
        for char in row:
            custom_map_flaten.append(char)
    # old_envP = copy.deepcopy(envP)
    old_envP = copy.copy(envP)

    new_envP = {}
    for state, v1 in old_envP.items():
        inner_dict = {}
        for action, v2 in v1.items():
            inner_list = []
            for tpl in v2:
                (prob_of_transition, s_prime, old_reward, terminated) = tpl
                if custom_map_flaten[s_prime] == "H":
                    new_reward = (
                        hole_reward  # Decrease the reward for falling into a hole
                    )
                elif custom_map_flaten[s_prime] == "G":
                    new_reward = (
                        goal_reward  # Increase the reward for reaching the goal
                    )
                else:
                    new_reward = move_reward  # Decrease the reward for movin
                inner_list.append((prob_of_transition, s_prime, new_reward, terminated))
            inner_dict[action] = inner_list
        new_envP[state] = inner_dict

    return new_envP


class ModifyRewards(gym.Wrapper):
    def __init__(
            self, env, custom_map, hole_reward=-10, goal_reward=10, move_reward=-0.1
    ):
        super().__init__(env)
        self.hole_reward = hole_reward
        self.goal_reward = goal_reward
        self.move_reward = move_reward
        self.custom_map = custom_map
        self.P = modify_rewards_P(
            env.P, custom_map, hole_reward, goal_reward, move_reward
        )

    def step(self, action):
        next_state, reward, done, truncated, info = self.env.step(action)

        modified_reward = modify_rewards(
            next_state,
            self.custom_map,
            self.hole_reward,
            self.goal_reward,
            self.move_reward,
        )
        return next_state, modified_reward, done, truncated, info


############################### plot methods : you can use them to plot 
# your policy and state value


#  plot policy with arrows in four direction to understand policy better
def plot_policy_arrows(policy, custom_map):
    custom_map_flaten = get_flaten_custom_map(custom_map)
    n = len(custom_map)
    m = len(custom_map[0])
    fig, ax = plt.subplots(n, m, figsize=(8, 8))
    for i in range(n):
        for j in range(m):
            ax[i, j].set_xlim([0, 3])
            ax[i, j].set_ylim([0, 3])
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
    for state, subdict in policy.items():
        row = state // m
        col = state % m
        if custom_map_flaten[state] == "S":
            square_fill = plt.Rectangle(
                (0.5, 0.5), 2, 2, linewidth=0, edgecolor=None, facecolor="y", alpha=0.5
            )
            ax[row, col].add_patch(square_fill)
        for direction, value in subdict.items():
            dx, dy = 0, 0
            if direction == 0:
                dx = -value
            elif direction == 1:
                dy = -value
            elif direction == 2:
                dx = value
            else:
                dy = value
            if value != 0:
                ax[row, col].arrow(1.5, 1.5, dx, dy, head_width=0.35, head_length=0.25)
        if subdict[0] == 0 and subdict[1] == 0 and subdict[2] == 0 and subdict[3] == 0:
            if custom_map_flaten[state] == "G":
                color = "g"
            else:
                color = "r"
            square_fill = plt.Rectangle(
                (0.5, 0.5),
                2,
                2,
                linewidth=0,
                edgecolor=None,
                facecolor=color,
                alpha=0.5,
            )
            ax[row, col].add_patch(square_fill)
    plt.show()


# plot policy in terminal using best action for each state
def plot_policy_terminal(policy, custom_map):
    arr = np.empty((len(custom_map), len(custom_map[0])), dtype=object)
    state = 0
    for i in range(len(custom_map)):
        for j in range(len(custom_map[i])):
            subdict = policy[state]
            best_action = max(subdict, key=subdict.get)
            if best_action == 0:
                arr[i, j] = "Lt"  # 0: LEFT
            elif best_action == 1:
                arr[i, j] = "Dn"  # 1: DOWN
            elif best_action == 2:
                arr[i, j] = "Rt"  # 2: RIGHT
            elif best_action == 3:
                arr[i, j] = "UP"  # 3: UP
            else:
                arr[i, j] = "##"
            state += 1
    print(arr)


# plot state value
def plot_state_value(state_value, custom_map):
    custom_map_flaten = get_flaten_custom_map(custom_map)
    rows = len(custom_map)
    cols = len(custom_map[0])
    table = state_value.reshape(rows, cols)
    # Define custom colors
    green = mcolors.to_rgba("green", alpha=0.5)
    blue = mcolors.to_rgba("blue", alpha=0.5)
    fig, ax = plt.subplots()
    im = ax.imshow(table, cmap="Reds")
    state = 0
    for i in range(rows):
        for j in range(cols):
            if custom_map_flaten[state] == "H":
                ax.add_patch(
                    mpatches.Rectangle(
                        xy=(j - 0.5, i - 0.5),
                        width=1,
                        height=1,
                        linewidth=0.1,
                        facecolor=blue,
                    )
                )
            elif custom_map_flaten[state] == "G":
                ax.add_patch(
                    mpatches.Rectangle(
                        xy=(j - 0.5, i - 0.5),
                        width=1,
                        height=1,
                        linewidth=0,
                        facecolor=green,
                    )
                )

            ax.text(
                j,
                i,
                str(i * cols + j) + "\n" + custom_map_flaten[state],
                ha="center",
                va="center",
            )
            state += 1
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xticklabels([""] * cols)
    ax.set_yticklabels([""] * rows)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Value", rotation=-90, va="bottom")
    ax.set_title("state value")
    plt.show()


############################### handler methods : you don't need to know them,
# they have been used in other methods


def act_wrt_prob(probability):
    if random.random() < probability:
        return 1
    else:
        return 0


def get_action_wrt_policy(state, policy):
    action = -1
    while action == -1:
        if act_wrt_prob(policy[state][0]) == 1:
            action = 0
        elif act_wrt_prob(policy[state][1]) == 1:
            action = 1
        elif act_wrt_prob(policy[state][2]) == 1:
            action = 2
        elif act_wrt_prob(policy[state][3]) == 1:
            action = 3
    return action


def get_flaten_custom_map(custom_map):
    custom_map_flaten = []
    for row in custom_map:
        for char in row:
            custom_map_flaten.append(char)
    return custom_map_flaten


############################### helper methods : you can use them in your code to create
# random policy and check your policy


# it gives a randome walk policy w.r.t costum 
def get_init_policy(custom_map):
    policy = {}
    random_sub_dict = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
    terminal_sub_dict = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    for i in range(len(custom_map)):
        for j in range(len(custom_map[i])):
            state = (i * len(custom_map[i])) + j
            if custom_map[i][j] == "H" or custom_map[i][j] == "G":

                policy[state] = terminal_sub_dict
            else:
                policy[state] = random_sub_dict

    return policy


# it gives walk policy according to direction w.r.t costum
def get_policy_direction(direction, custom_map):  # direction :"left", "down", "right"
    policy = {}
    left_sub_dict = {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0}
    down_sub_dict = {0: 0.0, 1: 1.0, 2: 0.0, 3: 0.0}
    right_sub_dict = {0: 0.0, 1: 0.0, 2: 1.0, 3: 0.0}
    terminal_sub_dict = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    for i in range(len(custom_map)):
        for j in range(len(custom_map[i])):
            state = (i * len(custom_map[i])) + j
            if custom_map[i][j] == "H" or custom_map[i][j] == "G":

                policy[state] = terminal_sub_dict
            else:
                if direction == "left":
                    policy[state] = left_sub_dict
                elif direction == "down":
                    policy[state] = down_sub_dict
                elif direction == "right":
                    policy[state] = right_sub_dict

    return policy


# it run game according to given policy
def do_policy(env, policy, episdoes=5):
    # episdoes = 10
    for ep in range(episdoes):
        n_state = env.reset()[0]
        done = False
        rewards = 0
        moves = 0
        while done is False:
            action = get_action_wrt_policy(n_state, policy)
            n_state, reward, done, truncated, info = env.step(action)
            rewards += reward
            moves += 1
        print("rewards:", rewards, " - moves:", moves, " - final state:", n_state)
    print(env.render())




def policy_iteration(env, custom_map, max_ittr=30, theta=0.01, discount_factor=0.9):
      # it gives a random-walk policy
    V = np.zeros(env.observation_space.n)
    # for i in range(len(custom_map)):
    #     for j in range(len(custom_map[i])):
    #         if custom_map[i][j] == "G":
    #             V[i*len(custom_map) +j]=100

    nS=env.observation_space.n
    policy=get_init_policy(custom_map)
    ittr = 0
    policy_stable = False
    while not policy_stable and ittr < max_ittr:
        # policy evaluation
        policy_evaluation(V ,policy,nS ,theta)
        policy = improvePolicy(env, V, discount_factor)


        ittr += 1
    print("finish")

    return V, policy

def eval_state_action(V, s, a, gamma=0.90):
    sum=0
    for p, next_s, rew, _ in env.P[s][a]:
        sum+=p * (rew + gamma*V[next_s])
        # print("suuuuuum",s,a,env.P[s][a])

    return np.sum([p * (rew + gamma*V[next_s]) for p, next_s, rew, _ in env.P[s][a]])

def policy_evaluation(V, policy,states, eps=0.01):
    '''
    Policy evaluation. Update the value function until it reach a steady state
    '''

    while True:
        delta = 0
        # loop over all states
        for state in range(states):
            old_v = V[state]
            # update V[s] using the Bellman equation
            sum=0
            for action in range(4):
                print("pol:",policy[state][action])
                # test=eval_state_action(V, state, action)*policy[state][action]
                test=eval_state_action(V, state, action)*policy[state][action]
                sum+=test
            V[state] = sum
            delta = max(delta, np.abs(old_v - V[state]))

        if delta < eps:
            break
    print(V)

def improvePolicy(env, valueFunctionVector, discountRate):
    import numpy as np
    # this matrix will store the q-values (action value functions) for every state
    # this matrix is returned by the function
    qvaluesMatrix = np.zeros((env.observation_space.n, 4))
    # this is the improved policy
    # this matrix is returned by the function
    improvedPolicy = {}

    for stateIndex in range(env.observation_space.n):
        action_dict= {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        # computes a row of the qvaluesMatrix[stateIndex,:] for fixed stateIndex,
        # this loop iterates over the actions

        for actionIndex in range(4):
            # computes the Bellman equation for the action value function
            for probability, nextState, reward, isTerminalState in env.P[stateIndex][actionIndex]:
                qvaluesMatrix[stateIndex][ actionIndex] = qvaluesMatrix[stateIndex][ actionIndex] + probability * (
                            reward + discountRate * valueFunctionVector[nextState])

        bestActionIndex = np.where(qvaluesMatrix[stateIndex, :] == np.max(qvaluesMatrix[stateIndex, :]))

        # form the improved policy
        # print("ki",stateIndex ,bestActionIndex[0][0])
        for i in range(np.size(bestActionIndex)):
            action_dict[bestActionIndex[0][i]] = 1 / np.size(bestActionIndex)


        improvedPolicy[stateIndex] =action_dict
    # print("improved",improvedPolicy)
    return improvedPolicy


def first_visit_mc_prediction(env, policy, num_episodes, gamma=1):
    # sum of returns for every state
    stateNumber = env.observation_space.n
    return_sum = np.zeros(stateNumber)
    state_count = np.zeros(stateNumber)
    v = np.zeros(stateNumber)

    for indexEpisode in range(num_episodes):
        # this list stores visited states in the current episode
        visitedStatesInEpisode = []
        # this list stores the return in every visited state in the current episode
        rewardInVisitedState = []
        (currentState, prob) = env.reset()
        visitedStatesInEpisode.append(currentState)
        print("Simulating episode {}".format(indexEpisode))
        ###########################################################################
        # START episode simulation
        ###########################################################################
        while True:
            action = get_action_wrt_policy(currentState, policy) #action in current state accoeding to policy
            (currentState, currentReward, terminalState, _, _) = env.step(action)
            rewardInVisitedState.append(currentReward)
            # if the current state is NOT terminal state
            if not terminalState:
                visitedStatesInEpisode.append(currentState)
            else:
                break
        # how many states we visited in an episode
        numberOfVisitedStates = len(visitedStatesInEpisode)

        # this is G=R_{t+1}+\gamma R_{t+2} + \gamma^2 R_{t+3} + ...
        G = 0
        for indexCurrentState in range(numberOfVisitedStates - 1, -1, -1):
            # summing the returns backwards
            stateTmp = visitedStatesInEpisode[indexCurrentState]
            returnTmp = rewardInVisitedState[indexCurrentState]
            G = gamma * G + returnTmp

            if stateTmp not in visitedStatesInEpisode[0:indexCurrentState]:
                #this state is visited in the episode
                state_count[stateTmp] = state_count[stateTmp] + 1
                # add the sum for that state to the total sum for that state
                return_sum[stateTmp] = return_sum[stateTmp] + G

    for indexSum in range(stateNumber):
        if state_count[indexSum] != 0:
            v[indexSum] = return_sum[indexSum] / state_count[indexSum]

    return v


# This algorithm allows you to estimate the state values of a given policy by sampling episodes and
# calculating the average returns(in every visit of a state)
def every_visit_mc_prediction(env, policy, num_episodes, gamma):
    # initilize
    V = np.zeros(env.observation_space.n)
    N = np.zeros(env.observation_space.n)

    # loop in range num_episodes(for each episode)
    # for i_episode in range(num_episodes):

    # generate episode w.r.t policy

    # loop for each step of episode , t= T-1, T-2, ..., 0




    # Initialize empty dictionaries to store returns and state counts
    returns_sum = {}
    returns_count = {}

    # Initialize state-value function V as a dictionary of zeros
    V = {state: 0 for state in range(env.observation_space.n)}

    # Loop over episodes
    for episode in range(num_episodes):
        # Generate an episode using the given policy
        states_visited = []
        rewards_received = []
        state = env.reset()
        print(state[0])
        done = False
        while not done:
            for i in range(4):
                print("i:",i ,policy[state[0]][i])
                if policy[state[0]][i]==1:
                    action=i
                    break

            next_state, reward, done ,truncated, info = env.step(action)
            states_visited.append(state)
            rewards_received.append(reward)
            state = next_state

        # Compute returns for each visited state in the episode
        G = 0
        for t in range(len(states_visited) - 1, -1, -1):
            G = gamma * G + rewards_received[t]
            if states_visited[t] in returns_sum:
                returns_sum[states_visited[t]] += G
                returns_count[states_visited[t]] += 1
            else:
                returns_sum[states_visited[t]] = G
                returns_count[states_visited[t]] = 1

        # Update state-value function V based on the computed returns
        for state in returns_sum:
            V[state] = returns_sum[state] / returns_count[state]

    return V


############################### custom maps : you have to use them according to the problem

custom_map_1 = ["HFSFFFFG"]

custom_map_2 = ["SFFFF",
                "HHHFF",
                "FFFFH",
                "FFFFF",
                "FFFFG"]

custom_map_3 = ["SFFFF",
                "HFFFF",
                "HFFFF",
                "HFFFF",
                "GFFFF"]

custom_map_4 = ["FFFSFFF",
                "FHHHHFF",
                "FFFFFFF",
                "HFFFFFF",
                "FGFFFFF"]

custom_map_5 = ["HFSFFFFG"]

custom_map_6 = ["HFSFFFFG",
                "HFFFFFFF",
                "HFFFFFFF"]

custom_map_7 = ["SFFFF",
                "FFFFH",
                "HHFFF",
                "HFFFH",
                "FFFFG"]

custom_map_8 = ["HFFSFFH",
                "FFFFFFF",
                "FFFFFFF",
                "GFFHFFG"]
#############################
if __name__ == "__main__":
    map = custom_map_7
    env = gym.make("FrozenLake-v1", render_mode="ansi", desc=map, is_slippery=False)
    # env = gym.make("FrozenLake-v1", desc=map, is_slippery=True)
    env = ModifyRewards(
        env, custom_map=map, hole_reward=-0.2, goal_reward=1, move_reward=-0.1
    )
    env.reset()
    print(env.render())
    ###
    v ,policy = policy_iteration(env, map, 50)
    v=first_visit_mc_prediction(env ,policy ,50 ,0.9)
    print("monto",v.reshape((len(map), len(map[0]))))
    print(policy)


    # policy = get_init_policy(map)
    # plot_policy_arrows(policy, map)


    do_policy(env, policy,5)

    rewards = 0
    for t in range(100):
        action = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(action)
        rewards += reward



        # action = 2
        # next_state, reward, done, truncated, info = env.step(action)
        # rewards += reward
        # action = 1
        # next_state, reward, done, truncated, info = env.step(action)
        # rewards += reward
        # action = 2
        # next_state, reward, done, truncated, info = env.step(action)
        # rewards += reward
        if done:
            print(rewards)
            break

    # V, policy = policy_iteration(env, map, theta=0.0001,discount_factor=0.99)

    # plot_state_value(v, map)
    # plot_policy_arrows(policy, map)
    plot_policy_terminal(policy, map)
    # s = 12
    # print(env.P[s][0])
    # print(env.P[s][1])
    # print(env.P[s][2])
    # print(env.P[s][3])
    # do_policy(env, policy, episdoes=5)

    # num_episodes = 10000
    # gamma = 0.9
    # V_MC = first_visit_mc_prediction(env, policy, num_episodes, gamma)

    # plot_state_value(V_MC, map)
    time.sleep(2)

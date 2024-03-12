from DSA import DSA
from MGM import MGM
from MailBox import MailBox
import random
import matplotlib.pyplot as plt



# create neighbor matrix
def matrix():
    w, h = 10, 10
    Matrix = [[0 for x in range(w)] for y in range(h)]
    for i in range(10):
        for j in range(10):
            cost = int(random.uniform(0, 99))
            Matrix[i][j] = cost
            Matrix[j][i] = cost
    return Matrix

# Create a list of the cost for the agent, for each value
# then, calculate the best value for agent
def best_choice(mails, agent):
    cost_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for m in mails:
        if m.to_id == agent.id_agent and m.iteration == agent.iteration:
            for n in agent.neighbors:
                if n['agent'] == m.from_id:
                    matrix = n['matrix']
                    for i in range(10):
                        cost_list[i] += matrix[m.value_from][i]
            mails.remove(m)
    min = 3001
    new_val = 0
    for x in cost_list:
        if x < min:
            min = x
            new_val = cost_list.index(x)
    return new_val


# Calculate the LR for the agent and add it to the agent
# also add the value to change if the agent has the best LR
def best_LR_choice(mails, agent):
    cost_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for m in mails:
        if m.to_id == agent.id_agent and m.iteration == agent.iteration:
            for n in agent.neighbors:
                if n['agent'] == m.from_id:
                    matrix = n['matrix']
                    for i in range(10):
                        cost_list[i] += matrix[m.value_from][i]
            mails.remove(m)
    min = 3001
    new_val = 0
    for x in cost_list:
        if x < min:
            min = x
            new_val = cost_list.index(x)

    LR = cost_list[agent.value] - cost_list[new_val]
    agent.set_LR(LR)
    agent.set_potential_val(new_val)


# For the DSA agent, change the value of agent based on p
def should_i_change(agent, val, p):
    rand = random.random()
    if rand < p:
        agent.set_value(val)


# Add message to mailbox for DSA agent
def send_message(agent, mb):
    for neighbor in agent.neighbors:
        message_to_mail = agent.create_msg(neighbor['agent'])
        mb.add_message(message_to_mail)


# Add mesage to mailbox for MGM agent
def send_LR_message(agent, mb):
    for neighbor in agent.neighbors:
        message_to_mail = agent.create_LR_msg(neighbor['agent'])
        mb.add_message(message_to_mail)


# Calculate the cost of the "System"
def total_cost(agents):
    final_values = []
    final_cost = 0
    for agent in agents:
        final_values.append(agent.value)
    for agent in agents:
        for n in agent.neighbors:
            n_matrix = n['matrix']
            n_value = final_values[n['agent']]
            final_cost += n_matrix[n_value][agent.value]
    return final_cost / 2


# Create DSA agents and run the system with a specific p and seed
def first_question(p, seed):
    random.seed(seed)
    k = 0.2
    agents = []
    mb = MailBox()
    iteration = 0
    num_of_iterations = 100
    final_values = []
    final_cost = 0

    # initialize agents
    for i in range(30):
        agents.append(DSA(i))

    # initialize neighbors and neighbor matrix
    for i in agents:
        for j in agents:
            if i.id_agent < j.id_agent:
                rand = random.random()
                if rand < k:
                    m = matrix()
                    i.add_neighbor({"agent": j.id_agent, "matrix": m})
                    j.add_neighbor({"agent": i.id_agent, "matrix": m})

    # first iteration
    for agent in agents:
        send_message(agent, mb)

    # all other iterations
    while iteration < num_of_iterations:
        iteration += 1
        for agent in agents:
            best = best_choice(mb.messages, agent)
            should_i_change(agent, best, p)
            agent.set_iteration(iteration)
            send_message(agent, mb)

    return total_cost(agents)

def run_seeds_and_average(p):
    sum = 0
    for i in range(30):
        sum += first_question(p, i)
    average = sum/30
    return average

def first_graph(p):
    x = []
    y = []
    while p <= 1.01:
        x.append(p)
        y.append(run_seeds_and_average(p))
        p += 0.05

    # plotting the points
    plt.plot(x, y)

    # naming the x axis
    plt.xlabel('p')
    # naming the y axis
    plt.ylabel('Average cost')

    # giving a title to my graph
    plt.title('First graph')

    # function to show the plot
    plt.show()


# Create DSA agents and run the system with a specific p, k and seed
def second_question(p, k, seed):
    random.seed(seed)
    agents = []
    mb = MailBox()
    iteration = 0
    num_of_iterations = 100
    x_axis = []
    y_axis = []

    # initialize agents
    for i in range(30):
        agents.append(DSA(i))

    # initialize neighbors and neighbor matrix
    for i in agents:
        for j in agents:
            if i.id_agent < j.id_agent:
                rand = random.random()
                if rand < k:
                    m = matrix()
                    i.add_neighbor({"agent": j.id_agent, "matrix": m})
                    j.add_neighbor({"agent": i.id_agent, "matrix": m})

    # first iteration
    for agent in agents:
        send_message(agent, mb)
    final_cost = 0
    final_values = []
    for agent in agents:
        final_values.append(agent.value)
    for agent in agents:
        for n in agent.neighbors:
            n_matrix = n['matrix']
            n_value = final_values[n['agent']]
            final_cost += n_matrix[n_value][agent.value]
    y_axis.append(final_cost / 2)
    x_axis.append(iteration)

    # all other iterations
    while iteration < num_of_iterations:
        iteration += 1
        for agent in agents:
            best = best_choice(mb.messages, agent)
            should_i_change(agent, best, p)
            agent.set_iteration(iteration)
            send_message(agent, mb)
        # For each iteration calculate the cost of the system
        final_cost = 0
        final_values = []
        for agent in agents:
            final_values.append(agent.value)
        for agent in agents:
            for n in agent.neighbors:
                n_matrix = n['matrix']
                n_value = final_values[n['agent']]
                final_cost += n_matrix[n_value][agent.value]
        y_axis.append(final_cost / 2)
        x_axis.append(iteration)

    return x_axis, y_axis


# check if the agent has better lr than his neighbors
def mgm_am_i_the_best(agent, mails):
    flag = True
    for m in mails.messages:
        if m.to_id == agent.id_agent and m.iteration == agent.iteration:
            if m.lr > agent.LR:
                flag = False
            if m.lr == agent.LR:
                if agent.id_agent > m.from_id:
                    flag = False
            mails.remove_message(m)
    return flag


# Create MGM agents and run the system with a specific k and seed
def mgm_question(k, seed):
    random.seed(seed)
    agents = []
    mb = MailBox()
    mb2 = MailBox()
    iteration = 0
    num_of_iterations = 100
    x_axis = []
    y_axis = []

    # initialize agents
    for i in range(30):
        agents.append(MGM(i))

    # initialize neighbors and neighbor matrix
    for i in agents:
        for j in agents:
            if i.id_agent < j.id_agent:
                rand = random.random()
                if rand < k:
                    m = matrix()
                    i.add_neighbor({"agent": j.id_agent, "matrix": m})
                    j.add_neighbor({"agent": i.id_agent, "matrix": m})

    # first iteration
    for agent in agents:
        send_message(agent, mb)

    final_cost = 0
    final_values = []
    for agent in agents:
        final_values.append(agent.value)
    for agent in agents:
        for n in agent.neighbors:
            n_matrix = n['matrix']
            n_value = final_values[n['agent']]
            final_cost += n_matrix[n_value][agent.value]
    y_axis.append(final_cost / 2)
    x_axis.append(iteration)


    # all other iterations
    while iteration < num_of_iterations:
        iteration += 1
        for agent in agents:
            best_LR_choice(mb.messages, agent)
            send_LR_message(agent, mb2)
            flag = mgm_am_i_the_best(agent, mb2)
            if flag:
                agent.value = agent.potential_val
            agent.set_iteration(iteration)
            send_message(agent, mb)


        final_cost = 0
        final_values = []
        for agent in agents:
            final_values.append(agent.value)
        for agent in agents:
            for n in agent.neighbors:
                n_matrix = n['matrix']
                n_value = final_values[n['agent']]
                final_cost += n_matrix[n_value][agent.value]
        y_axis.append(final_cost / 2)
        x_axis.append(iteration)

    return x_axis, y_axis


# run question 2 on 30 seeds on the DSA agents
def run_seeds_and_average_for_2(p, k):
    y_axis = [0] * 101
    for i in range(30):
        x, y = second_question(p, k, i)
        for j in x:
            y_axis[j] += y[j]
    for i in range(len(y_axis)):
        y_axis[i] = y_axis[i]/30
    return x, y_axis

# run question 2 on 30 seeds on the MGM agents
def run_seeds_and_average_for_mgm(k):
    y_axis = [0] * 101
    for i in range(30):
        x, y = mgm_question(k, i)
        for j in x:
            y_axis[j] += y[j]
    for i in range(len(y_axis)):
        y_axis[i] = y_axis[i]/30
    return x, y_axis


def second_graph():
    x_axis, y_axis1 = run_seeds_and_average_for_2(0.2, 0.2)
    x_axis, y_axis2 = run_seeds_and_average_for_2(0.7, 0.2)
    x_axis, y_axis3 = run_seeds_and_average_for_mgm(0.2)


    # plotting the points
    plt.plot(x_axis, y_axis1, color='y',label='0.2')
    plt.plot(x_axis, y_axis2, color='g',label='0.7')
    plt.plot(x_axis, y_axis3, color='b', label='mgm')

    # naming the x axis
    plt.xlabel('Iteration')
    # naming the y axis
    plt.ylabel('Cost')

    # giving a title to my graph
    plt.title('Second graph')

    plt.legend()

    # function to show the plot
    plt.show()


def third_graph():
    x_axis, y_axis1 = run_seeds_and_average_for_2(0.2, 0.7)
    x_axis, y_axis2 = run_seeds_and_average_for_2(0.7, 0.7)
    x_axis, y_axis3 = run_seeds_and_average_for_mgm(0.7)

    # plotting the points
    plt.plot(x_axis, y_axis1, color='y', label='0.2')
    plt.plot(x_axis, y_axis2, color='g', label='0.7')
    plt.plot(x_axis, y_axis3, color='b', label='mgm')

    # naming the x axis
    plt.xlabel('Iteration')
    # naming the y axis
    plt.ylabel('Cost')

    # giving a title to my graph
    plt.title('Third graph')

    plt.legend()

    # function to show the plot
    plt.show()


first_graph(0)
second_graph()
third_graph()


import gym
import pickle
import numpy as np
from copy import deepcopy
from multiprocessing import Pool

MAX_SEED = 2**16 - 1



def _check_remove_node_out(sub_elite, node, outg, incm):
    """ Recursively remove connecting incoming
       and outgoing nodes if ~either~ outgoing or incoming empty """
    remove_count = 0
    if len(incm) > 0:
        for _node in incm:
            sub_elite["nodes"][_node]["outgoing"].remove(node)
            remove_count += 1
            if len(sub_elite["nodes"][_node]["outgoing"]) == 0 \
                    or len(sub_elite["nodes"][_node]["incoming"]) == 0:
                if not (_node in sub_elite["input"] or _node in sub_elite["output"]):
                    sub_elite, remove_cp = _check_remove_node_out(
                        sub_elite, _node,
                        sub_elite["nodes"][_node]["outgoing"],
                        sub_elite["nodes"][_node]["incoming"]
                    )
                    del sub_elite["nodes"][_node]
                    remove_cp += remove_count
    if len(outg) > 0:
        for _node in outg:
            sub_elite["nodes"][_node]["incoming"].remove(node)
            remove_count += 1
            if len(sub_elite["nodes"][_node]["outgoing"]) == 0 \
                    or len(sub_elite["nodes"][_node]["incoming"]) == 0:
                if not (_node in sub_elite["input"] or _node in sub_elite["output"]):
                    sub_elite, remove_cp = _check_remove_node_out(
                        sub_elite, _node,
                        sub_elite["nodes"][_node]["outgoing"],
                        sub_elite["nodes"][_node]["incoming"]
                    )
                    remove_count += 1
                    del sub_elite["nodes"][_node]
                    remove_cp += remove_count

    return sub_elite, remove_count



def remove_weight(sub_elite, conn, decay_type="simple"):
    """
    Remove weight from graph
    :param sub_elite:
    :param conn:
    :param decay_type:
     [Types] full, simple, incoming_del, outgoing_del
    :return:
    """
    remove_count = 0
    sub_elite["nodes"][conn[0]]["outgoing"].remove(conn[1])
    sub_elite["nodes"][conn[1]]["incoming"].remove(conn[0])
    if sub_elite["weight_optim"]:
        del sub_elite["nodes"][conn[0]]["weights"][conn[1]]
    remove_count += 1
    if decay_type == "full":
        """ Recursively remove connecting incoming 
           and outgoing nodes if either outgoing or incoming empty """
        if (not (conn[0] in sub_elite["input"] or conn[0] in sub_elite["output"])
            and (len(sub_elite["nodes"][conn[0]]["outgoing"]) == 0
                 or len(sub_elite["nodes"][conn[0]]["incoming"]) == 0)):
            """ Remove connections to conn[0] """
            sub_elite, remove_upd = _check_remove_node_out(
                sub_elite, conn[0],
                sub_elite["nodes"][conn[0]]["outgoing"],
                sub_elite["nodes"][conn[0]]["incoming"]
            )
            del sub_elite["nodes"][conn[0]]
            remove_count += remove_upd
        if (not (conn[1] in sub_elite["input"] or conn[1] in sub_elite["output"])
            and (len(sub_elite["nodes"][conn[1]]["outgoing"]) == 0
                 or len(sub_elite["nodes"][conn[1]]["incoming"]) == 0 )):
            """ Remove connections to conn[0] """
            sub_elite, remove_upd = _check_remove_node_out(
                sub_elite, conn[1],
                sub_elite["nodes"][conn[1]]["outgoing"],
                sub_elite["nodes"][conn[1]]["incoming"]
            )
            del sub_elite["nodes"][conn[1]]
            remove_count += remove_upd
    elif decay_type == "simple":
        """ Delete node if incoming and outgoing connections are dead """
        if not (conn[0] in sub_elite["input"] or conn[0] in sub_elite["output"]):
            if len(sub_elite["nodes"][conn[0]]["outgoing"]) == 0\
                    and len(sub_elite["nodes"][conn[0]]["incoming"]) == 0:
                del sub_elite["nodes"][conn[0]]
        if conn[0] != conn[1]:
            if not (conn[1] in sub_elite["input"] or conn[1] in sub_elite["output"]):
                if len(sub_elite["nodes"][conn[1]]["outgoing"]) == 0\
                        and len(sub_elite["nodes"][conn[1]]["incoming"]) == 0:
                    del sub_elite["nodes"][conn[1]]
    elif decay_type == "incoming_del":
        """ Delete node if incoming connections are dead, but not outgoing """
        if not (conn[0] in sub_elite["input"] or conn[0] in sub_elite["output"]):
            if len(sub_elite["nodes"][conn[0]]["incoming"]) == 0:
                tmp_sub_elite = deepcopy(sub_elite)
                for _node in sub_elite["nodes"][conn[0]]["outgoing"]:
                    tmp_sub_elite["nodes"][_node]["incoming"].remove(conn[0])
                sub_elite = deepcopy(tmp_sub_elite)
                del sub_elite["nodes"][conn[0]]
        if conn[0] != conn[1]:
            if not (conn[1] in sub_elite["input"] or conn[1] in sub_elite["output"]):
                if len(sub_elite["nodes"][conn[1]]["incoming"]) == 0:
                    tmp_sub_elite = deepcopy(sub_elite)
                    for _node in sub_elite["nodes"][conn[1]]["outgoing"]:
                        tmp_sub_elite["nodes"][_node]["incoming"].remove(conn[1])
                    sub_elite = deepcopy(tmp_sub_elite)
                    del sub_elite["nodes"][conn[1]]
        elif decay_type == "outgoing_del":
            """ Delete node if outgoing connections are dead, but not incoming """
            if not (conn[0] in sub_elite["input"] or conn[0] in sub_elite["output"]):
                if len(sub_elite["nodes"][conn[0]]["outgoing"]) == 0:
                    tmp_sub_elite = deepcopy(sub_elite)
                    for _node in sub_elite["nodes"][conn[0]]["incoming"]:
                        tmp_sub_elite["nodes"][_node]["outgoing"].remove(conn[0])
                    sub_elite = deepcopy(tmp_sub_elite)
                    del sub_elite["nodes"][conn[0]]
            if conn[0] != conn[1]:
                if not (conn[1] in sub_elite["input"] or conn[1] in sub_elite["output"]):
                    if len(sub_elite["nodes"][conn[1]]["outgoing"]) == 0:
                        tmp_sub_elite = deepcopy(sub_elite)
                        for _node in sub_elite["nodes"][conn[1]]["incoming"]:
                            tmp_sub_elite["nodes"][_node]["outgoing"].remove(conn[1])
                        sub_elite = deepcopy(tmp_sub_elite)
                        del sub_elite["nodes"][conn[1]]
    return sub_elite, remove_count


"""
todo: remove weight is slow
-- Make incoming and outgoing...
"""


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def sigmoid(x):
    return (np.tanh(x/2.0) + 1.0)/2.0


def relu(x):
    return np.maximum(0, x)


def identity(x):
    return x


def gaussian(x):
    return np.exp(-np.multiply(x, x) / 2.0)


def step(x):
    return 1.0 * (x>0.0)


ACTIVATIONS = {
    "tanh" : np.tanh,
    "id"   : identity,
    "sin"  : lambda x: np.sin(np.pi*x),
    "cos"  : lambda x: np.cos(np.pi*x),
    "sig"  : sigmoid,
    "step" : step,
    "clip" : lambda x: np.clip(x, -1, 1),
    "gaus" : gaussian,
    "relu" : relu,
    "inv"  : lambda x: -x,
}


# only using activations that work with plasticity
ACTIVATION_CHOICES = [
    "tanh",
    "id"  ,
    "sin" ,
    "cos" ,
    "sig" ,
    "step",
    "clip",
    "gaus",
    "relu",
    "inv",
]


OUTP_ACTIVATION_CHOICES = [
    "tanh",
    "id"  ,
    "sin" ,
    "cos" ,
    "sig" ,
    "step",
    "clip",
]


class NetworkGraph:
    def __init__(self, graph, rand_mutate_steps=1000, learn_weights=False, learn_weight_type="continuous"):
        self.graph = graph
        self.prev_graph = self.graph
        self.learn_weights = learn_weights
        self.learn_weight_type = learn_weight_type

        self.connections = 0
        for _node in self.graph['nodes']:
            self.connections += len(self.graph['nodes'][_node]['outgoing'])

        self.local_rew = -10000000
        self.prev_rew = self.local_rew

        self.iterations = 0
        self.update_steps = 0
        self.rand_mutate_steps = rand_mutate_steps
        self.rand_mutation_probs = [0.2, 0.2, 0.2, 0.2, 0.2]
        if self.learn_weights:
            self.graph["weight_optim"] = True
            if self.learn_weight_type == "continuous":
                self.rand_mutation_probs = [0.1, 0.1, 0.1, 0.1, 0.6]
            elif self.learn_weight_type == "discrete":
                self.rand_mutation_probs = [0.1, 0.1, 0.1, 0.1, 0.6]
            elif self.learn_weight_type == "binary":
                self.rand_mutation_probs = [0.2, 0.2, 0.2, 0.2, 0.2]
            elif self.learn_weight_type == "weight_share":
                self.rand_mutation_probs = [0.2, 0.2, 0.2, 0.2, 0.2]
                self.graph['shared_weight'] = 0.1
                self.graph["weight_optim"] = False
            elif self.learn_weight_type == "random_discrete":
                self.rand_mutation_probs = [0.25, 0.25, 0.25, 0.25, 0.0]
                self.graph["weight_optim"] = False
            elif self.learn_weight_type == "random_continuous":
                self.rand_mutation_probs = [0.25, 0.25, 0.25, 0.25, 0.0]
                self.graph["weight_optim"] = False

        else:
            self.rand_mutation_probs = [0.25, 0.25, 0.25, 0.25, 0.0]
            self.graph["weight_optim"] = False
        #self.rand_mutation_probs = [0.25, 0.25, 0.5, 0.0, 0.0]

    def forward(self, x):
        self.iterations += 1
        x = x.squeeze()
        prior_nodes = set()
        for _inp in self.graph['input']:
            self.graph['nodes'][_inp]['val'] += x[_inp]
        queue = deepcopy(self.graph['input'])
        while len(queue) > 0:
            node = queue[0]
            if node not in self.graph['output'] and len(self.graph['nodes'][node]['outgoing']) > 0:
                w_val = ACTIVATIONS[self.graph['nodes']
                    [node]['activation']](self.graph['nodes'][node]['val'])
                self.graph['nodes'][node]['val'] = 0
                for outgoing_node in self.graph['nodes'][node]['outgoing']:
                    if outgoing_node not in prior_nodes:
                        queue.append(outgoing_node)
                    prior_nodes.add(outgoing_node)
                    if self.learn_weights:
                        if self.learn_weight_type == "weight_share":
                            self.graph['nodes'][outgoing_node]['val'] += \
                                w_val*self.graph['shared_weight']
                        elif self.learn_weight_type == "random_discrete":
                            self.graph['nodes'][outgoing_node]['val'] += \
                                w_val * np.random.choice([-1.0, -0.5, -0.25, 0.25, 0.5, 1.0])
                        elif self.learn_weight_type == "random_continuous":
                            self.graph['nodes'][outgoing_node]['val'] += \
                                w_val * np.random.uniform(low=-1, high=1)
                        else:
                            self.graph['nodes'][outgoing_node]['val'] += \
                                w_val*self.graph['nodes'][node]['weights'][outgoing_node]
                    else:
                        self.graph['nodes'][outgoing_node]['val'] += w_val*1
            #else:
            #    self.graph['nodes'][node]['val'] = 0
            queue.pop(0)
        output = list()
        for _outp in self.graph['output']:
            w_val = ACTIVATIONS[self.graph['nodes']
                [_outp]['activation']](self.graph['nodes'][_outp]['val'])
            output.append(w_val)
            self.graph['nodes'][_outp]['val'] = 0
        output = np.clip(np.array(output), a_max=1, a_min=-1)
        return output + np.random.normal(loc=0.0, scale=0.01, size=output.size)

    def mutate(self, local_reward, rand_mutation_probs=None, num_modifications=None,):
        self.local_rew = local_reward

        if num_modifications is None:
            num_modifications = 1

        if self.rand_mutate_steps is not None:
            if self.local_rew >= self.prev_rew:
                self.prev_graph = deepcopy(self.graph)
            else:
                self.graph = deepcopy(self.prev_graph)
                self.local_rew = deepcopy(self.prev_rew)
            self.graph, updates = mutate(
                self.graph,
                num_modifications=num_modifications, #self.num_mutations,
                random_operation_probs=rand_mutation_probs,
                learn_weight_type=self.learn_weight_type
            )
            for _upd in updates:
                if _upd == "node":
                    self.connections += 1
                elif _upd == "addweight":
                    self.connections += 1
                elif type(_upd) == tuple:
                    self.connections -= _upd[1]

            self.graph["iteration"] += 1
            self.update_steps += 1
            self.prev_rew = deepcopy(self.local_rew)
            self.local_rew = 0.0

    def reset(self):
        self.iterations = 0
        self.update_steps = 0
        self.prev_rew = -10000000
        self.local_rew = -10000000


def mutate(sub_elite, num_modifications, random_operation_probs, learn_weight_type):
    updates = list()
    random_operations = ["addweight", "node", "activation", "remove_weight", "change_weight"]
    for _mod in range(num_modifications):
        operation = np.random.choice(random_operations, p=random_operation_probs)
        if operation == "activation":
            node_set = list((set(sub_elite["nodes"].keys())
                .difference(set(sub_elite["input"]))).difference(set(sub_elite["output"])))
            if len(node_set) == 0:
                sub_elite, _ = mutate(
                    sub_elite,
                    num_modifications=1,
                    random_operation_probs=[0.0, 1.0, 0.0, 0.0, 0.0],
                    learn_weight_type=learn_weight_type
                )
                updates.append("node")
                continue
            #for _try in range(10):
            node = np.random.choice(node_set)
            #if node in sub_elite["output"] or node in sub_elite["input"]: continue
            sub_elite["nodes"][node]["activation"] = np.random.choice(ACTIVATION_CHOICES)
            updates.append(operation)


        elif operation == "node":
            nodes = sub_elite["nodes"]
            _conns = [(nodes[_node]["outgoing"], _node)
                for _node in nodes if len(nodes[_node]["outgoing"]) > 0]
            _connections = list()
            for _connection in _conns:
                for _node in _connection[0]:
                    _connections.append((_connection[1], _node))
            if len(_connections) == 0:
                sub_elite, _ = mutate(
                    sub_elite,
                    num_modifications=1,
                    random_operation_probs=[1.0, 0.0, 0.0, 0.0, 0.0],
                    learn_weight_type=learn_weight_type
                )
                updates.append("addweight")
                continue
            _conn = _connections[np.random.choice(list(range(len(_connections))))]
            _outgoing, _receiving = _conn[0], _conn[1]
            _new_node_id = sub_elite["next_node_id"]
            if sub_elite["weight_optim"]:
                _old_weight = deepcopy(sub_elite["nodes"][_outgoing]["weights"][_receiving])
            sub_elite["next_node_id"] += 1
            sub_elite["nodes"][_new_node_id] = {
                "val": 0,
                "activation": "tanh", #np.random.choice(ACTIVATION_CHOICES),
                "outgoing": {_receiving},
                "incoming": {_outgoing },
                "weights": {} if not sub_elite["weight_optim"] else {_receiving: 1.0}
            }
            sub_elite["nodes"][_outgoing]["outgoing"].remove(_receiving)
            sub_elite["nodes"][_receiving]["incoming"].remove(_outgoing)
            if sub_elite["weight_optim"]:
                del sub_elite["nodes"][_outgoing]["weights"][_receiving]
            sub_elite["nodes"][_outgoing]["outgoing"].add(_new_node_id)
            sub_elite["nodes"][_receiving]["incoming"].add(_new_node_id)
            if sub_elite["weight_optim"]:
                sub_elite["nodes"][_outgoing]["weights"][_new_node_id] = _old_weight
            updates.append(operation)

        elif operation == "addweight":
            for _try in range(10):
                node1 = np.random.choice(list(sub_elite["nodes"].keys()))
                node2 = np.random.choice(list(sub_elite["nodes"].keys()))
                if node2 in sub_elite["input"] or \
                        (node1 == node2 and node1 in sub_elite["output"]):
                    continue
                elif node2 not in sub_elite["output"] and \
                        len(sub_elite["nodes"][node2]['outgoing']) == 0:
                    continue
                sub_elite["nodes"][node1]['outgoing'].add(node2)
                sub_elite["nodes"][node2]['incoming'].add(node1)
                if learn_weight_type == "continuous":
                    sub_elite["nodes"][node1]["weights"][node2] = np.random.normal(loc=0.0, scale=0.02) #np.random.choice([-1.0, -0.5, -0.1, 0.1, 0.5, 1.0])
                elif learn_weight_type == "discrete":
                    sub_elite["nodes"][node1]["weights"][node2] = np.random.choice([-1.0, -0.5, -0.25, 0.25, 0.5, 1.0])
                elif learn_weight_type == "binary":
                    sub_elite["nodes"][node1]["weights"][node2] = np.random.choice([-1.0, 1.0])
                updates.append(operation)
                break

        elif operation == "remove_weight":
            nodes = sub_elite["nodes"]
            _conns = [(nodes[_node]["outgoing"], _node)
                      for _node in nodes if len(nodes[_node]["outgoing"]) > 0]
            _connections = list()
            for _connection in _conns:
                for _node in _connection[0]:
                    _connections.append((_connection[1], _node))
            if len(_connections) == 0:
                sub_elite, _ = mutate(
                    sub_elite,
                    num_modifications=1,
                    random_operation_probs=[1.0, 0.0, 0.0, 0.0, 0.0],
                    learn_weight_type=learn_weight_type
                )
                updates.append("addweight")
                continue
            _conn = _connections[np.random.choice(list(range(len(_connections))))]
            sub_elite, remove_count = remove_weight(sub_elite, _conn)
            updates.append((operation, remove_count))

        elif operation == "change_weight":
            nodes = sub_elite["nodes"]
            _conns = [(nodes[_node]["outgoing"], _node)
                      for _node in nodes if len(nodes[_node]["outgoing"]) > 0]
            _connections = list()
            for _connection in _conns:
                for _node in _connection[0]:
                    _connections.append((_connection[1], _node))
            if len(_connections) == 0:
                sub_elite, _ = mutate(
                    sub_elite,
                    num_modifications=1,
                    random_operation_probs=[1.0, 0.0, 0.0, 0.0, 0.0],
                    learn_weight_type=learn_weight_type
                )
                updates.append("addweight")
                continue
            _conn = _connections[np.random.choice(list(range(len(_connections))))]
            if learn_weight_type == "continuous":
                sub_elite["nodes"][_conn[0]]["weights"][_conn[1]] += np.random.normal(loc=0.0, scale=0.01)
            elif learn_weight_type == "discrete":
                sub_elite["nodes"][_conn[0]]["weights"][_conn[1]] = np.random.choice([-1.0, -0.5, -0.25, 0.25, 0.5, 1.0])
            elif learn_weight_type == "binary":
                sub_elite["nodes"][_conn[0]]["weights"][_conn[1]] = np.random.choice([-1.0, 1.0])
            elif learn_weight_type == "shared_weight":
                sub_elite['shared_weight'] += np.random.normal(loc=0.0, scale=0.02)
            updates.append(operation)

    return sub_elite, updates

"""
Only evolve architecture?
Update weights using N(0, s)?
Use gradients to update 
  W upon accepting mutation?

1) Learn weight agnostic architecture 
2) Learn architecture and weights (RL-grad) 
3) Learn architecture, weights, and morphology
4) Figure out how to reduce update interval

"""


"""

Penalize new connections in a weight agnositc setting
- disable a limb ONLINE and show that weight connections die off after N updates etc

"""


def open_ended_evolution(graph):
    sub_interval = 500
    update_interval = 2500
    best_reward = -10000000
    max_env_interacts = 10000000

    sub_itr = 0
    updates = 0
    save_rate = 1

    learn_weights = True
    penalize_connections = True
    learn_weight_type = "discrete" # || binary (-1, 1), continuous + N, discrete, weight_share

    return_avg = 0.0
    prev_trace = list()
    update_int_avg = list()
    update_int_max = list()
    momentary_reward = list()

    local_env = gym.make("AntLifelong-v2")
    network = NetworkGraph(
        deepcopy(graph),
        rand_mutate_steps=update_interval,
        learn_weights=learn_weights,
        learn_weight_type=learn_weight_type
    )
    network.reset()
    #network.mutate(
    #    network.local_rew+0.01,
    #    rand_mutation_probs=[0.3, 0.2, 0.0, 0.5, 0.0],
    #    num_modifications=50,
    #)
    network.prev_graph = deepcopy(network.graph)
    state = local_env.reset()
    sub_int = list()
    for _inter in range(max_env_interacts):
        state = state.reshape((1, state.size))
        state = np.concatenate((state, np.ones((1, 1))), axis=1)
        action1 = network.forward(state).squeeze()
        state, reward, game_over, _info = local_env.step(action1)
        # ~~~~~~~~ Online Reward ~~~~~~~~
        return_avg += reward
        momentary_reward.append(reward)
        if (sub_itr+1) % sub_interval == 0:
            sub_int.append(deepcopy(return_avg))
            return_avg = 0.0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if (sub_itr + 1) % update_interval == 0:
            sub_int = np.array(sub_int)
            sub_int.sort()
            return_avg = np.median(sub_int) - 0.1*abs(sum(sub_int[-2:])/2 - sum(sub_int[:2])/2)

            #scale = 250/update_interval

            avg_rew = return_avg#*scale
            if penalize_connections:
                avg_rew_penalty = return_avg #- network.connections*0.001 #*scale
                avg_rew = avg_rew_penalty

            if return_avg > best_reward: #*scale > best_reward:
                best_reward = return_avg#*scale
                with open("oenas_best.pkl", "wb") as f:
                    pickle.dump(network.graph, f)

            #std = np.array(running_average).std() + sum(running_average)/len(running_average)
            network.mutate(
                avg_rew,
                num_modifications=np.random.choice(list(range(1, 20))),
                rand_mutation_probs=network.rand_mutation_probs,
            )

            #if network.prev_rew > 0:
            #    if avg_rew < 0:
            #        avg_rew = 1
            network.prev_rew *= 0.99 #(np.log10(max(avg_rew/network.prev_rew, 0.01)*100000000000000000 + 0.1) + 1 )/18
            #else:
            #    network.prev_rew *= 1.01

            prev_trace.append(network.prev_rew)
            update_int_avg.append(sub_int.mean())#*scale)
            update_int_max.append(best_reward)

            if updates % save_rate == 0:
                with open("oenas_avg.pkl", "wb") as f:
                    pickle.dump(update_int_avg, f)
                with open("oenas_max.pkl", "wb") as f:
                    pickle.dump(update_int_max, f)
                with open("oenas_graph.pkl", "wb") as f:
                    pickle.dump(network.prev_graph, f)
                with open("oenas_ptrace.pkl", "wb") as f:
                    pickle.dump(prev_trace, f)
            print("Local: {}, Avg {}, Prev: {}, Graph Itr: {}, Inter: {}".format(
                round(avg_rew, 4),
                round(sub_int.mean(), 4),
                round(network.prev_rew, 4),
                network.graph["iteration"], _inter)
            )
            momentary_reward = list()
            return_avg = 0.0
            sub_itr = 0
            updates += 1
            sub_int = list()

        sub_itr += 1

    with open("oenas_avg.pkl", "wb") as f:
        pickle.dump(update_int_avg, f)
    with open("oenas_max.pkl", "wb") as f:
        pickle.dump(update_int_max, f)
    with open("oenas_graph.pkl", "wb") as f:
        pickle.dump(network.prev_graph, f)


if __name__ == "__main__":
    _env = gym.make("AntLifelong-v2")
    input_dim = _env.observation_space.shape[0]
    output_dim = _env.action_space.shape[0]
    input_nodes = list(range(input_dim))
    output_nodes = [_ + input_dim for _ in range(output_dim)]
    input_n = np.random.choice(
        input_nodes, replace=False, size=(np.random.randint(1, input_dim),))
    random_perm = set(input_nodes)
    _init_graph = {
        "input": input_nodes,
        "output": output_nodes,
        "nodes": {
            _: {"outgoing": set(),  # set(np.random.choice(
                # output_nodes,
                # replace=False,
                # size=(1,))) if _ in random_perm else set(),
                "incoming": set(),
                "activation": "id",
                "val": 0,
                "weights": dict(),
                }
            for _ in input_nodes
        },
        "next_node_id": input_dim + output_dim,
        "iteration": 0,
    }
    _init_graph["nodes"].update({
        _: {"outgoing": set(),
            "incoming": set(),  # set([_n for _n in _init_graph["nodes"] if _ in _init_graph["nodes"][_n]["outgoing"]]),
            "activation": "tanh",  # np.random.choice(OUTP_ACTIVATION_CHOICES),
            "val": 0,
            "weights": dict(),
            }
        for _ in output_nodes
    })
    open_ended_evolution(graph=_init_graph)






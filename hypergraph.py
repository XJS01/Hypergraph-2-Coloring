import networkx as nx
import matplotlib.pyplot as plt
import argparse
import numpy as np

np.random.seed(0)

class hypergraph:
    def __init__(self, num_nodes, edges):
        self.hypergraph = nx.Graph()
        self.hypergraph.add_nodes_from(np.arange(num_nodes))
        self.num_nodes = num_nodes
        self.num_edges = len(edges)
        self.node_colors = np.random.randint(2 ,size=self.num_nodes)
        for i, edge in enumerate(edges):
            edge_name = self.get_edge_name(i)
            self.hypergraph.add_node(edge_name)
            self.hypergraph.add_edges_from([(edge_name, node) for node in edge])

    def get_color_split(self):
        return np.where(self.node_colors == 0)[0], np.where(self.node_colors == 1)[0]

    def find_maximal_independent_set_of_monochromatic_edges(self):
        maximal_independent_set = []
        not_in_maximal_independent_set = []
        nodes_to_resample = []

        skip_edges = set()
        for i in range(self.num_edges):
            edge_name = self.get_edge_name(i)
            if edge_name in skip_edges:
                not_in_maximal_independent_set.append(edge_name)
                continue
            node_neighbors = list(self.hypergraph.neighbors(edge_name))
            color_sum = np.sum(np.take(self.node_colors, node_neighbors))
            if color_sum % self.hypergraph.degree[edge_name] == 0:
                nodes_to_resample.extend(node_neighbors)
                maximal_independent_set.append(edge_name)
                for node in self.hypergraph.neighbors(edge_name):
                    skip_edges.update(self.hypergraph.neighbors(node))
            else:
                not_in_maximal_independent_set.append(edge_name)
        return maximal_independent_set, nodes_to_resample, not_in_maximal_independent_set

    def reshuffle(self):
        maximal_independent_set, nodes_to_resample, _ = self.find_maximal_independent_set_of_monochromatic_edges()
        if len(maximal_independent_set) == 0:
            return False, 0
        self.node_colors[nodes_to_resample] = np.random.randint(2 ,size=len(nodes_to_resample))
        return True, len(maximal_independent_set)

    def parallel_LLL(self):
        total_num_events_resampled = 0
        shuffled, num_events_resampled = self.reshuffle()
        total_num_events_resampled += num_events_resampled
        while shuffled:
            shuffled, num_events_resampled = self.reshuffle()
            total_num_events_resampled += num_events_resampled
        return total_num_events_resampled

    def parallel_LLL_with_graph(self):
        total_num_events_resampled = 0
        self.plot_hypergraph("Initial Coloring")
        shuffled, num_events_resampled = self.reshuffle()
        total_num_events_resampled += num_events_resampled
        iteration = 0
        while shuffled:
            iteration += 1
            self.plot_hypergraph("Iteration {0}".format(iteration))
            shuffled, num_events_resampled = self.reshuffle()
            total_num_events_resampled += num_events_resampled
        return total_num_events_resampled

    def get_edge_name(self, i):
        return "e{0}".format(i)

    def plot_hypergraph(self, title):
        if self.num_nodes > 6 or self.num_edges > 6:
            node_size = 25
            edge_size = 10
            include_labels = False
        else:
            node_size = 1000
            edge_size = 500
            include_labels = True
        r = 5
        plt.figure()
        axes = plt.axes()
        plt.gca().set_xlim((-1.2*r, 1.2*r))
        plt.gca().set_ylim((-1.2*r, 1.2*r))
        plt.gca().set_aspect(1.0)
        node_angle_diff = 2*np.pi/self.num_nodes
        edge_angle_diff = 2*np.pi/self.num_edges
        node_positions = {i: (r * np.cos(node_angle_diff * i), r * np.sin(node_angle_diff * i)) for i in np.arange(self.num_nodes)}
        node_positions.update({self.get_edge_name(i): (0.5*r * np.cos(edge_angle_diff * i), 0.5*r * np.sin(edge_angle_diff * i)) for i in np.arange(self.num_edges)})
        fixed_nodes = list(range(self.num_nodes)) + [self.get_edge_name(i) for i in range(self.num_edges)]

        maximal_independent_set, _, not_in_maximal_indepenent_set = self.find_maximal_independent_set_of_monochromatic_edges()

        pos = nx.spring_layout(self.hypergraph, pos=node_positions, fixed=fixed_nodes)

        color0_nodes, color1_nodes = self.get_color_split()
        nx.draw_networkx_nodes(self.hypergraph, pos, node_size=node_size, nodelist=color0_nodes, ax=axes, node_color='#f77f00')
        nx.draw_networkx_nodes(self.hypergraph, pos, node_size=node_size, nodelist=color1_nodes, ax=axes, node_color='#0000ff')
        nx.draw_networkx_nodes(self.hypergraph, pos, node_size=edge_size, nodelist=not_in_maximal_indepenent_set, ax=axes, node_color='#adff2f')
        nx.draw_networkx_nodes(self.hypergraph, pos, node_size=edge_size, nodelist=maximal_independent_set, ax=axes, node_color='#d62828')
        nx.draw_networkx_edges(self.hypergraph, pos, edge_color='#eae2b7', connectionstyle='arc3,rad=0.05', arrowstyle='-')

        if include_labels:
            labels = {node: str(node) for node in self.hypergraph.nodes}
            nx.draw_networkx_labels(self.hypergraph, pos, labels)
        plt.title(title)
        plt.show()

def generate_random_hypergraph(num_nodes, num_edges, k):
    max_num_neighbors = int(2**(k-1)/np.e - 1)
    edge_to_edge_map = {edge :set() for edge in range(num_edges)}
    node_to_edge_map = {node :set() for node in range(num_nodes)}
    edges = []
    for i in range(num_edges):
        num_adj_nodes_remaining = k
        node_perm = np.random.permutation(num_nodes)
        perm_idx = 0
        edge = []
        while num_adj_nodes_remaining:
            curr_node = node_perm[perm_idx]
            curr_node_edges = node_to_edge_map[curr_node]
            curr_edge_edges = edge_to_edge_map[i]
            new_edges_added = 0
            perm_idx += 1
            flag = False
            for node_edge in curr_node_edges:
                is_new = i not in edge_to_edge_map[node_edge]
                new_edges_added += is_new
                if len(edge_to_edge_map[node_edge]) > max_num_neighbors - is_new:
                    flag = True
                    break
            if flag:
                continue
            if len(curr_edge_edges) + new_edges_added <= max_num_neighbors:
                for node_edge in curr_node_edges:
                    edge_to_edge_map[node_edge].add(i)
                    edge_to_edge_map[i].add(node_edge)
                curr_node_edges.add(i)
                edge.append(curr_node)
                num_adj_nodes_remaining -= 1
        edges.append(edge)
    return hypergraph(num_nodes, edges), max_num_neighbors

def visualize_constructive_LLL():
    visualization_graph, _ = generate_random_hypergraph(132, 40, 4)
    visualization_graph.parallel_LLL_with_graph()

def plot_runtime_k():
    num_resampled_list = []
    upper_bounds_in_expectation_list = []
    k_range = np.arange(5, 13)
    num_edges = 1500
    for k in k_range:
        g, d = generate_random_hypergraph(5000, num_edges, k)
        num_resampled = g.parallel_LLL()
        num_resampled_list.append(num_resampled)
        x = 1/(d + 1)
        upper_bounds_in_expectation_list.append(num_edges * (x / (1 - x)))

    plt.figure()
    plt.plot(k_range, num_resampled_list, label="Number of Events Resampled")
    plt.plot(k_range, upper_bounds_in_expectation_list, label="Upper Bound in Expectation")
    plt.title("Number of Events Resampled vs. k")
    plt.xlabel("k")
    plt.ylabel("Number of Events")
    plt.legend()
    plt.show()

def plot_runtime_edges():
    num_resampled_list = []
    upper_bounds_in_expectation_list = []
    edges_range = np.arange(1, 256)
    for i in edges_range:
        g, d = generate_random_hypergraph(500, i, 7)
        num_resampled = g.parallel_LLL()
        num_resampled_list.append(num_resampled)
        x = 1/(d + 1)
        upper_bounds_in_expectation_list.append(i * (x / (1-x)))

    plt.figure()
    plt.plot(edges_range, num_resampled_list, label="Number of Events Resampled")
    plt.plot(edges_range, upper_bounds_in_expectation_list, label="Upper Bound in Expectation")
    plt.title("Number of Events Resampled vs. Number of Edges")
    plt.xlabel("Number of Edges")
    plt.ylabel("Number of Events")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualization", "-v", type=str)
    args = parser.parse_args()
    if args.visualization == "k":
        plot_runtime_k()
    elif args.visualization == "edges":
        plot_runtime_edges()
    elif args.visualization == "hypergraph":
        visualize_constructive_LLL()

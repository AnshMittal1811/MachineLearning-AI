__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '3/24/2020 10:59 AM'

import itertools
import sys
import time
import random

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession


def cmp(key1, key2):
    """

    :param key1:
    :param key2:
    :return:
    """
    return (key1, key2) if key1 < key2 else (key2, key1)


def export2File(result_array, file_path):
    """
    export list content to a file
    :param result_array: a list of dict
    :param file_path: output file path
    :return: nothing, but a file
    """
    with open(file_path, 'w+') as output_file:
        for id_array in result_array:
            output_file.writelines(str(id_array)[1:-1] + "\n")
        output_file.close()


def update_dict(dict_obj, key, increment):
    """
    update the value with the same key, rather than replace it
    :param dict_obj:
    :param key:
    :param increment:
    :return:
    """
    old_weight = dict_obj[key]
    dict_obj[key] = float(old_weight + increment)
    return dict_obj


def extend_dict(dict_obj, increment_dict):
    """
    same as list extend
    :param dict_obj:
    :param increment_dict:
    :return:
    """
    for key, value in increment_dict.items():
        if key in dict_obj.keys():
            dict_obj = update_dict(dict_obj, key, value)
        else:
            dict_obj[key] = value
    return dict_obj


class GraphFrame(object):

    def __init__(self, vertexes, edges):
        """

        :param vertexes: list of vertexes [1,2,3,4...]
        :param edges: a big dict(vertex: (list of vertex it connected)
        """
        self.vertexes = vertexes
        self.vertex_weight_dict = dict()
        self.__init_weight_dict__()

        self.edges = edges
        self.__init_adjacent_matrix__(edges)

        # variable using for compute betweenness
        self.betweenness_result_dict = dict()
        self.betweenness_result_tuple_list = None

        # variable using for compute modularity
        self.best_communities = None

    def __init_weight_dict__(self):
        [self.vertex_weight_dict.setdefault(vertex, 1) for vertex in self.vertexes]

    def __init_adjacent_matrix__(self, edges):
        """
        build a set which contain all edge pair
        :param edges: original edges (a big dict (vertex: [list of vertex it connected]))
        :return:
        """
        self.original_edges = edges
        self.m = self._count_edges(edges)

        # build adjacent matrix for original edges
        edge_set = set()
        for start_node, end_nodes in edges.items():
            for end_node in end_nodes:
                edge_set.add(cmp(start_node, end_node))
        self.A_matrix = edge_set

    def _count_edges(self, edges):
        """
        :param edges:  a big dict(vertex: (list of vertex it connected)
        :return:
        """
        visited = set()
        count = 0
        for start_node, end_nodes in edges.items():
            for end_node in end_nodes:
                key = cmp(start_node, end_node)
                if key not in visited:
                    visited.add(key)
                    count += 1
        return count

    def _build_tree(self, root):
        # root set in level 0 and no parent
        tree = dict()
        tree[root] = (0, list())

        # since BFS only visit each node once,
        # so use visited variable to save these records
        visited = set()

        need2visit = list()
        need2visit.append(root)

        while len(need2visit) > 0:
            parent_node = need2visit.pop(0)
            visited.add(parent_node)
            for children in self.edges[parent_node]:
                if children not in visited:
                    visited.add(children)
                    tree[children] = (tree[parent_node][0] + 1, [parent_node])
                    need2visit.append(children)
                elif tree[parent_node][0] + 1 == tree[children][0]:
                    tree[children][1].append(parent_node)

        return {k: v for k, v in sorted(tree.items(), key=lambda kv: -kv[1][0])}

    def _traverse_tree(self, tree_dict):
        """
        traverse the tree and compute weight for each edge
        :param tree_dict: {'2GUjO7NU88cPXpoffYCU8w': (9, ['a48HhwcmjFLApZhiax41IA']), ...
        :return:
        """
        weight_dict = self.vertex_weight_dict.copy()
        shortest_path_dict = self._find_num_of_paths(tree_dict)
        result_dict = dict()
        for key, value in tree_dict.items():
            if len(value[1]) > 0:
                denominator = sum([shortest_path_dict[parent] for parent in value[1]])
                for parent in value[1]:
                    temp_key = cmp(key, parent)
                    contribution = float(float(weight_dict[key]) * int(shortest_path_dict[parent]) / denominator)
                    result_dict[temp_key] = contribution
                    # update every parent node weight
                    weight_dict = update_dict(weight_dict, parent, contribution)

        return result_dict

    def _find_num_of_paths(self, tree_dict):
        """
        find how many the number of shortest path each node has
        :param tree_dict: {'2GUjO7NU88cPXpoffYCU8w': (9, ['a48HhwcmjFLApZhiax41IA']), ...
        :return: {'y6jsaAXFstAJkf53R4_y4Q': 1, '0FVcoJko1kfZCrJRfssfIA': 1, '2quguRdKBzul ...
        """
        level_dict = dict()
        shortest_path_dict = dict()
        for child_node, level_parents in tree_dict.items():
            level_dict.setdefault(level_parents[0], []) \
                .append((child_node, level_parents[1]))

        for level in range(0, len(level_dict.keys())):
            for (child_node, parent_node_list) in level_dict[level]:
                if len(parent_node_list) > 0:
                    shortest_path_dict[child_node] = sum([shortest_path_dict[parent]
                                                          for parent in parent_node_list])
                else:
                    shortest_path_dict[child_node] = 1
        return shortest_path_dict

    def computeBetweenness(self):
        """
        compute betweenness of each edge pair
        :return: list of tuple(pair, float)
                => e.g. [(('0FVcoJko1kfZCrJRfssfIA', 'bbK1mL-AyYCHZncDQ_4RgA'), 189.0), ...
        """
        self.betweenness_result_dict = dict()
        for node in self.vertexes:
            # 1.The algorithm begins by performing a breadth-first search
            # (BFS) of the graph, starting at the vertex X in all vertexes list
            # =>{'2GUjO7NU88cPXpoffYCU8w': (9, ['a48HhwcmjFLApZhiax41IA']),
            # '6YmRpoIuiq8I19Q8dHKTHw': (9, ['a48Hh
            bfs_tree = self._build_tree(root=node)
            # 2. Label each node by the number of shortest
            # paths that reach it from the root node
            # actually, this step has been done in the first step,
            # since the len of value[1] is exactly the number of shortest path
            # 3. Calculate for each edge e, the sum over all nodes
            # Y (of the fraction) of the shortest paths from the root
            # X to Y that go through edge e
            temp_result_dict = self._traverse_tree(bfs_tree)

            self.betweenness_result_dict = extend_dict(self.betweenness_result_dict,
                                                       temp_result_dict)

        # 4. Divide by 2 to get true betweenness
        self.betweenness_result_dict = \
            dict(map(lambda kv: (kv[0], float(kv[1] / 2)),
                     self.betweenness_result_dict.items()))

        self.betweenness_result_tuple_list = sorted(
            self.betweenness_result_dict.items(), key=lambda kv: (-kv[1], kv[0][0]))

        return self.betweenness_result_tuple_list

    def extractCommunities(self):
        """
        extract communities from butch of edge pairs
        :return:
        """
        max_modularity = float("-inf")
        # reuse the betweenness dict
        if len(self.betweenness_result_tuple_list) > 0:
            # cut edges with highest betweenness
            self._cut_highest_btw_edge(self.betweenness_result_tuple_list)
            self.best_communities, max_modularity = self._computeModularity()
            # recompute and update self.betweenness_result_tuple_list
            self.betweenness_result_tuple_list = self.computeBetweenness()

        while True:
            # cut edges with highest betweenness
            self._cut_highest_btw_edge(self.betweenness_result_tuple_list)
            communities, current_modularity = self._computeModularity()
            self.betweenness_result_tuple_list = self.computeBetweenness()
            if current_modularity >= max_modularity:
                # when current_modularity > max_modularity happens:
                # we still need to cut the edges
                self.best_communities = communities
                max_modularity = current_modularity
                print("current_modularity -> ", current_modularity)
                print("communities ->", communities)

            if current_modularity == 0:
                break

        return sorted(self.best_communities, key=lambda item: (len(item), item[0], item[1]))

    def _cut_highest_btw_edge(self, edge_btw_tuple_list):
        """
        remove edges with highest betweenness and also update the self.edges
        :param edge_btw_tuple_list: need to be a [sorted] list, sorted by value
        :return:
        """
        # this is the edge you need to cut
        temp_value = 0
        # if there have multiple pair have same highest bet score,
        # we cut them in one loop
        edge_pair = edge_btw_tuple_list[0][0]

        if self.edges[edge_pair[0]] is not None:
            try:
                self.edges[edge_pair[0]].remove(edge_pair[1])
            except ValueError:
                pass

        if self.edges[edge_pair[1]] is not None:
            try:
                self.edges[edge_pair[1]].remove(edge_pair[0])
            except ValueError:
                pass

    def _computeModularity(self):
        """
        compute the modularity based on communities we get
        :return: a list of communities and a float number => modularity
        """

        # 1. detect communities from current edge_pairs
        communities = self._detectCommunities()

        # 2. compute modularity based on the communities
        # 2.1 count original graph's edge number => self.m
        # 2.2 build adjacent matrix => self.A_matrix
        temp_sum = 0
        for cluster in communities:
            for node_pair in itertools.combinations(list(cluster), 2):
                temp_key = cmp(node_pair[0], node_pair[1])
                k_i = len(self.edges[node_pair[0]])
                k_j = len(self.edges[node_pair[1]])
                A = 1 if temp_key in self.A_matrix else 0
                temp_sum += float(A - (k_i * k_j / (2 * self.m)))
        return communities, float(temp_sum / (2 * self.m))

    def _detectCommunities(self):
        """
        detect communities based on self.edge
        basically, we randomly pick one root and find all connected node with root
        and then do the same thing on the rest of node
        :return: a list of set() which contain communities
        """
        communities = list()  # result will be return
        need2visit = list()  # a stack actually
        temp_node_set = set()  # using to save each communities
        visited = set()  # track which node has been visited

        # random pick a root to detect communities
        random_root = self.vertexes[random.randint(0, len(self.vertexes) - 1)]
        temp_node_set.add(random_root)
        need2visit.append(random_root)
        # if still has some node we haven't visit, do the loop
        while len(visited) != len(self.vertexes):
            while len(need2visit) > 0:
                parent_node = need2visit.pop(0)
                temp_node_set.add(parent_node)
                visited.add(parent_node)
                for children in self.edges[parent_node]:
                    if children not in visited:
                        temp_node_set.add(children)
                        need2visit.append(children)
                        visited.add(children)

            communities.append(sorted(temp_node_set))
            temp_node_set = set()
            if len(self.vertexes) > len(visited):
                # pick one from rest of unvisited nodes
                need2visit.append(set(self.vertexes).difference(visited).pop())

        return communities


if __name__ == '__main__':
    start = time.time()
    # define input variables
    filter_threshold = "7"
    input_csv_path = "../data/ub_sample_data.csv"
    betweenness_file_path = "../out/task2_bet4.txt"
    community_file_path = "../out/task2_com4.txt"

    # filter_threshold = sys.argv[1]
    # input_csv_path = sys.argv[2]
    # betweenness_file_path = sys.argv[3]
    # community_file_path = sys.argv[4]

    conf = SparkConf().setMaster("local") \
        .setAppName("ay_hw_4_task2") \
        .set("spark.executor.memory", "4g") \
        .set("spark.driver.memory", "4g")
    sc = SparkContext(conf=conf)
    sparkSession = SparkSession(sc)
    sc.setLogLevel("WARN")

    # read the original json file and remove the header
    raw_data_rdd = sc.textFile(input_csv_path)
    header = raw_data_rdd.first()
    uid_bidxes_dict = raw_data_rdd.filter(lambda line: line != header) \
        .map(lambda line: (line.split(',')[0], line.split(',')[1])) \
        .groupByKey().mapValues(lambda bids: sorted(list(bids))) \
        .collectAsMap()

    uid_pairs = list(itertools.combinations(list(uid_bidxes_dict.keys()), 2))

    edge_list = list()
    vertex_set = set()
    for pair in uid_pairs:
        if len(set(uid_bidxes_dict[pair[0]]).intersection(
                set(uid_bidxes_dict[pair[1]]))) >= int(filter_threshold):
            edge_list.append(tuple(pair))
            edge_list.append(tuple((pair[1], pair[0])))
            vertex_set.add(pair[0])
            vertex_set.add(pair[1])

    # => ['B7IvZ26ZUdL2jGbYsFVGxQ', 'jnn504CkjtfbYIwBquWmBw', 'sBqCpEUn0qYdpSF4Db
    vertexes = sc.parallelize(sorted(list(vertex_set))).collect()

    # => {'39FT2Ui8KUXwmUt6hnwy-g': ['0FVcoJko1kfZCrJRfssfIA', '1KQi8Ym
    edges = sc.parallelize(edge_list).groupByKey() \
        .mapValues(lambda uidxs: sorted(list(set(uidxs)))).collectAsMap()

    graph_frame = GraphFrame(vertexes, edges)
    betweenness_result = graph_frame.computeBetweenness()
    # export your finding
    export2File(betweenness_result, betweenness_file_path)

    communities_result = graph_frame.extractCommunities()
    # export your finding
    export2File(communities_result, community_file_path)

    print("Duration: %d s." % (time.time() - start))

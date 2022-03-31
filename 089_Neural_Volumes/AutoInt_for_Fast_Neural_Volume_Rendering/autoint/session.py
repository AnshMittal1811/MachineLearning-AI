import networkx as nx
import matplotlib.pyplot as plt
from .autograd_modules import AutoDiffNode, Input, Constant
import copy
import itertools
from colour import Color
from networkx.drawing.nx_pydot import graphviz_layout


# A session represents a computation graph: we can compute on it and calculate its backward graph to
# obtain a new session. A session is also an AutoDiffNode
# --------------------
# Typical life of a session object:
# 0) A session is attached to a model that uses AutoDiffNodes (eg. SIREN in modules)
# 1) A session instantiates a forward computation graph when running forward() on a model
#    consisting of AutoDiffNodes
# 2) One can create the backward computation graph calling the get_backward_graph() function
#    on the session
# 3) One can compute the session using the compute_graph() function
class Session(AutoDiffNode):
    def __init__(self):
        super().__init__()
        # This is the forward graph
        self.G = nx.DiGraph()
        # This is a stack of nodes that are being processed
        # in the forward computation
        self.previous_nodes = []

        # order in topological sort to resolve ambiguities
        self.topo_order_resolution = {}

        # to uniquely identify a session (mostly debug)
        self.uid = hash(self)
        # index to uniquely name parameters as they are added
        self.param_idx = 0

        print(f"Creating session: {self.uid}")
        self.sorted_graph = None

    def find_root(self):
        nodes_with_zero_in_degree = [node for node, degree in self.G.in_degree() if degree == 0]
        if len(nodes_with_zero_in_degree) == 1:  # there is more than 1 node with in-degree=0 (that cannot be a tree)!
            return nodes_with_zero_in_degree[0]

    # Adds a node uniquely identified by 'module', with attribute 'name',in the forward graph
    def add_node_forward(self, module, inputs_to_module, name):
        if not self.G.has_node(module):
            self.G.add_node(module, nname=name,
                            copy_from=None,
                            ncolor=Color(pick_for=module))

            # assign a unique name to the module itself
            module.name = name.replace('\n', '_') + '_' + f'{self.param_idx:02d}' + '_' + f'{self.uid}'
            self.add_module(module.name, module)
            module.order_idx = self.param_idx
            self.param_idx += 1

            if self.previous_nodes:
                for input_module in inputs_to_module.values():
                    # this checks that the input module has been added before in
                    # the session (otherwise it raises an error
                    input_idx = self.previous_nodes.index(input_module)
                    # then we pop the input
                    new_input = self.previous_nodes.pop(input_idx)
                    # and add it to the graph
                    self.G.add_edge(module, new_input)
            self.previous_nodes.append(module)

    # Creates a list of all the forward subgraphs for each of the forward input to 'starting_node'
    def _get_forward_input_subgraph(self, starting_node, back_session):
        # Traverse the whole tree uniquely and copy it (because we care about deep copies of module,
        # we cannot just copy the graph) for each of the children input of the starting node
        input_graph_list = []
        root_node_list = []
        for src_node in self.G.successors(starting_node):  # for each of the input to a node
            input_graph = nx.DiGraph()

            # a dictionnary that maintains for each node in the original graph a pointer to its copy
            # in the input_graph
            org_to_copy_dict = dict()
            root_node = None  # the root node of the input_graph

            for node in list(nx.dfs_preorder_nodes(self.G, source=src_node)):
                new_node = copy.deepcopy(node)
                org_to_copy_dict.update({node: new_node})

                # We chain the 'copy_from': that is we want copy_from to point to the original node
                # in case it is not original (original node has copy_from=None)
                org_copy_from = self.G.nodes[node]['copy_from']
                if org_copy_from is None:
                    org_copy_from = node
                input_graph.add_node(new_node, nname=self.G.nodes[node]['nname'],
                                     copy_from=org_copy_from,
                                     ncolor=Color(pick_for=org_copy_from))

                new_node.name = new_node.get_label().replace('\n', '_') + '_' + f'{back_session.param_idx:02d}' + '_' + f'{back_session.uid}'
                back_session.add_module(new_node.name, new_node)
                new_node.order_idx = back_session.param_idx
                back_session.param_idx += 1

                nx.set_node_attributes(self.G, {node: Color(pick_for=node)}, 'ncolor')  # update the color of the origin
                if (root_node is None):
                    root_node = new_node
                else:
                    parent_in_org_graph = next(self.G.predecessors(node))  # G.predecessors is an iterator (we use next), there is a single predecessor, (tree)
                    parent_in_cpy_graph = org_to_copy_dict[parent_in_org_graph]
                    input_graph.add_edge(parent_in_cpy_graph, new_node)
                    parent_in_cpy_graph.set_input_at_pos(new_node, parent_in_org_graph.get_input_pos(node))

            input_graph_list.append(input_graph)
            root_node_list.append(root_node)
        return input_graph_list, root_node_list

    # Gets the n-th forward input of 'node'
    def _get_forward_input(self, node, n):
        inputs = self.G.successors(node)  # returns an iterator

        # use itertools.islice to apply n times the iterator,
        # and use next to get the first element
        return next(itertools.islice(inputs, n, n+1))

    # Parse the backward expression provided by a node to create the autodiff tree
    def _parse_backward_expr(self, expr, root, calling_node, back_session):
        # for all the elements in the expression
        for n_elem, elem in enumerate(expr):
            if isinstance(elem, list):
                self._parse_backward_expr(elem, root, calling_node, back_session)
            else:
                # Elem is a module: it is a leaf
                if isinstance(elem, AutoDiffNode):
                    # An element, add the node
                    name = elem.get_label()
                    back_session.G.add_node(elem, nname=name,
                                            copy_from=None,
                                            ncolor=Color(pick_for=calling_node))  # elem makes the color unique, calling node makes all the derived elem the same color

                    # this element was generated in a call to gen_backward and
                    # so will automatically share weights from the homologous model
                    # in the forward session
                    elem.name = name.replace('\n', '_') + '_' f'{back_session.param_idx:02d}' + '_' f'{back_session.uid}'
                    back_session.add_module(elem.name, elem)
                    elem.order_idx = back_session.param_idx
                    back_session.param_idx += 1

                    # and connect it to the level above
                    if root is not None:
                        back_session.G.add_edge(root, elem)
                        root.add_input(elem)
                    if elem != expr[-1] and isinstance(expr[n_elem + 1], list):  # if not the last element and the next element
                        root = elem  # is a list, then i'm the new root (the list are the children)
                # Elem is a string: it is a leaf
                # It can be either a derivative id (eg. 'd0') or an input number (eg. '0')
                elif isinstance(elem, str):
                    if elem[0] == 'd':
                        der_input_id = int(elem[1:])
                        input = self._get_forward_input(calling_node, der_input_id)
                        # Recurse: create the derivative for that input
                        self._create_backward_graph(back_session, input, root)
                    else:
                        input_id = int(elem[0:])
                        input_graph_list, input_graph_root_list = self._get_forward_input_subgraph(calling_node, back_session)
                        # add the parameters in the module
                        # map(back_session.add_module,input_graph_list)
                        # create the input graph, and add it
                        input_graph = input_graph_list[input_id]
                        back_session.G.add_nodes_from(input_graph.nodes(data=True))
                        back_session.G.add_edges_from(input_graph.edges)
                        # connect it to the level above
                        if root is not None:  # Not sure this could ever be 'None'?
                            back_session.G.add_edge(root, input_graph_root_list[input_id])
                            root.add_input(input_graph_root_list[input_id])

    def get_backward_graph(self):
        new_session = Session()
        self._create_backward_graph(new_session)
        return new_session

    # Creates the backward graph for 'forward_node' anchored at 'backward_root_node' in back_session
    def _create_backward_graph(self, back_session, forward_node=None, backward_root_node=None):
        if forward_node is None:
            # forward_node = self.previous_nodes[0] # this is the last node processed in the forward
            #                                       pass, hence, that is its root node
            forward_node = self.find_root()
        backward_expr = forward_node.gen_backward()
        self._parse_backward_expr(backward_expr, backward_root_node, forward_node, back_session)

    def preprocess(self):
        self.sorted_graph = []
        for node in nx.topological_sort(self.G.reverse(copy=False)):
            self.sorted_graph.append(node)

    def compute_graph_fast(self, model_input):
        partial_results = []
        reusable_results = {}

        if self.sorted_graph is None:
            node_generator = nx.topological_sort(self.G.reverse(copy=False))
        else:
            node_generator = self.sorted_graph
        for node in node_generator:
            num_inputs = node.get_num_inputs()
            inputs = [partial_results.pop(0) for i in range(num_inputs)]  # get a list of inputs
            # if the node was copied to somewhere then we should check whether we can reuse some computation

            if isinstance(node, Input) or isinstance(node, Constant):
                if node.id in model_input:
                    node.set_value(model_input[node.id])
                elif 'Constant' in node.name:
                    pass
                else:
                    raise ValueError(f"Input argument dictionary does not contain "
                                     f"entry corresponding to an Input or Constant "
                                     f"module with id {node.id}")

            if node.copied_from in reusable_results:
                output = reusable_results[node.copied_from]
            else:
                # we run the _forward()--/!\ not forward() /!\-- because we do not want to build the graph here
                output = node._forward(inputs)

            partial_results.insert(0, output)

            # If this node has been copied from another (in the forward graph),
            # then let us store the result using the key it comes from (makes it unique despite having many copies)
            if node.copied_from:
                if node.copied_from not in reusable_results:
                    reusable_results.update({node.copied_from: output})

        return partial_results[0]

    def compute_graph_fast2(self, model_input):
        partial_results = {}
        reusable_results = {}

        if self.sorted_graph is None:
            node_generator = nx.topological_sort(self.G.reverse(copy=False))
        else:
            node_generator = self.sorted_graph
        for node in node_generator:
            input_node_list = node.get_inputs()

            inputs = [partial_results[input_node] for input_node in input_node_list.values()]  # get a list of inputs

            if isinstance(node, Input) or isinstance(node, Constant):
                if node.id in model_input:
                    node.set_value(model_input[node.id])
                elif 'Constant' in node.name:
                    pass
                else:
                    raise ValueError(f"Input argument dictionary does not contain "
                                     f"entry corresponding to an Input or Constant "
                                     f"module with id {node.id}")

            # if the node was copied to somewhere then we should check whether we can reuse some computation
            if node.copied_from in reusable_results:
                output = reusable_results[node.copied_from]
            else:
                # we run the _forward()--/!\ not forward() /!\-- because we do not want to build the graph here
                output = node._forward(inputs)

            partial_results.update({node: output})

            # If this node has been copied from another (in the forward graph),
            # then let us store the result using the key it comes from (makes it unique despite having many copies)
            if node.copied_from:
                if node.copied_from not in reusable_results:
                    reusable_results.update({node.copied_from: output})

        return output

    # Computes an output from the graph
    def compute_graph(self, model_input):
        partial_results = []

        for node in nx.topological_sort(self.G.reverse(copy=False)):
            num_inputs = node.get_num_inputs()
            inputs = [partial_results.pop(0) for i in range(num_inputs)]  # get a list of inputs

            if isinstance(node, Input) or isinstance(node, Constant):
                if node.id in model_input:
                    node.set_value(model_input[node.id])
                elif 'Constant' in node.name:
                    pass
                else:
                    raise ValueError(f"Input argument dictionary does not contain "
                                     f"entry corresponding to an Input or Constant "
                                     f"module with id {node.id}")

            output = node.forward(inputs)
            partial_results.insert(0, output)

        return output

    # Returns the computation graph as a nested list of modules
    def to_nested_list(self, parent_node):
        list_rep = [parent_node]
        for node in self.G.successors(parent_node):
            list_rep.append(self.to_nested_list(node))
        return list_rep

    # Draws the graph in a session
    def draw(self, title=''):
        def get_node_pos_tree(graph):
            # the trick is to relabel the nodes with an integer since pydot gets confused
            # with nodes that have the same name (pydot does not work like networkx that
            # uses hashes, pydot needs unique names)
            new_graph = nx.convert_node_labels_to_integers(graph, label_attribute='nname')
            new_graph_pos = graphviz_layout(new_graph, prog='dot')
            graph_pos = {new_graph.nodes[n]['nname']: p for n, p in new_graph_pos.items()}
            return graph_pos

        def get_node_colors(graph):
            # color_list = nx.get_node_attributes(graph, 'ncolor') # problem here is that we want a list!
            _, color_list = list(zip(*graph.nodes(data='ncolor')))
            color_list_hex = [color.hex_l for color in color_list]
            return color_list_hex

        pos = get_node_pos_tree(self.G)
        labels = nx.get_node_attributes(self.G, 'nname')
        colors = get_node_colors(self.G)

        plt.figure()
        plt.title(('Graph: '+title+f" {self.uid}").title())

        nx.draw(self.G, pos, node_color=colors, alpha=0.5, with_labels=False)
        nx.draw_networkx_labels(self.G, pos, labels, font_size=9)

    ''' These functions make the session an AutoDiffNode '''
    def get_num_inputs(self):
        mode = 'all'

        all_input_list = [node for node in list(self.G) if isinstance(node, Input)]
        if(mode == 'all'):
            num_inputs = len(all_input_list)
        elif(mode == 'unique'):
            # maybe we want to return the number of unique inputs, in that case we need to
            # check which copy_from is unique
            copy_from_list = nx.get_node_attributes(self.G, 'copy_from')  # get the copy_from for all nodes
            all_input_copy_from_list = [copy_from_list[node] for node in all_input_list]  # filter nodes in the copy_from list that are inputs
            unique_input_list = list(set(all_input_copy_from_list))  # select only the unique ones
            num_inputs = len(unique_input_list)
        return num_inputs

    def gen_backward(self):
        back_session = self.get_backward_graph()
        return back_session.to_nested_list(back_session.find_root())

    def _forward(self, x):
        return self.compute_graph(x)

    def forward(self, model_input):
        out = self.compute_graph_fast2(model_input)
        return out

    def simplify(self, children=None):
        for node_to_simplify in nx.topological_sort(self.G.reverse(copy=False)):
            children = [child for child in self.G.successors(node_to_simplify)]

            action, replacing_node = node_to_simplify.simplify(children)
            if action == 'do_nothing':
                pass
            elif action == 'replace_by':
                parent_of_node_to_replace = next(self.G.predecessors(node_to_simplify))
                if not self.G.has_node(replacing_node):
                    self.G.add_node(replacing_node, nname=replacing_node.get_label(),
                                    copy_from=None,
                                    ncolor=Color(pick_for=replacing_node))
                self.G.add_edge(parent_of_node_to_replace, replacing_node)
                self.G.remove_node(node_to_simplify)  # removes node AND adjacent edges
        return 'do_nothing', None

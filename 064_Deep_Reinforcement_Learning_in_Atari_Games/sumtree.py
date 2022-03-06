import numpy as np 
# store the priorities of experience
# the priorities of memory samples are stored in leaf node, and the value of parent node is the sum of its children
class Sumtree():
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2*capacity - 1) # store the priorities of memory
        self.tree[capacity - 1] = 1
        self.stored = [False] * (2*capacity - 1) # indicate whether this node is used to store
        # self.cur_point = 0 
        self.length = 0 # maximum length is capacity
        self.push_count = 0

    def update_node(self, index, change):
        # update sum tree from leaf node if the priority of leaf node changed
        parent = (index-1)//2
        self.tree[parent] += change
        self.stored[parent] = True
        if parent > 0:
            self.update_node(parent, change)

    def update(self, index_memory, p):
        # update sum tree from new priority
        index = index_memory + self.capacity - 1
        change = p - self.tree[index]
        self.tree[index] = p
        self.stored[index] = True
        self.update_node(index, change)

    def get_p_total(self):
        # return total priorities
        return self.tree[0]

    def get_p_min(self):
        return min(self.tree[self.capacity-1:self.length+self.capacity-1])

    def get_by_priority(self, index, s):
        # get index of node by priority s
        left_child = index*2 + 1
        right_child = index*2 + 2
        if left_child >= self.tree.shape[0]:
            return index
        if self.stored[left_child] == False:
            return self.get_by_priority(right_child, s-self.tree[left_child])
        if self.stored[right_child] == False:
            return self.get_by_priority(left_child, s)
        if s <= self.tree[left_child]:
            return self.get_by_priority(left_child, s)
        else:
            return self.get_by_priority(right_child, s-self.tree[left_child])

    def sample(self, s):
        # sample node by priority s, return the index and priority of experience
        self.stored[self.length + self.capacity - 2] = False # cannot sample the latest state
        index = self.get_by_priority(0, s)
        return index - self.capacity + 1, self.tree[index]

    def push(self):
        # push experience, the initial priority is the maximum priority in sum tree
        index_memory = self.push_count % self.capacity
        if self.length < self.capacity:
            self.length += 1
        self.update(index_memory, np.max(self.tree[self.capacity-1 : self.capacity+self.length-1]))
        self.push_count += 1

    



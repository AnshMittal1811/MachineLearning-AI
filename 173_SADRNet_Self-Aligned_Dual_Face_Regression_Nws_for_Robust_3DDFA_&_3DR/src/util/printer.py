class DecayVarPrinter:
    def __init__(self, beta=0.99, warm_step=100):
        self.data = dict()
        self.beta = beta
        self.warm_step = warm_step

    def add_variable(self, key, init_val=0):
        self.data[key] = [init_val, 1]

    def update_variable_decay(self, key, new_val):
        if key not in self.data.keys():
            self.add_variable(key, init_val=new_val)
            return

        x, step = self.data[key]
        if step < self.warm_step:
            new_step = step + 1
            new_x = (x * step + new_val) / new_step
            self.data[key] = [new_x, new_step]
        else:
            new_step = step + 1
            new_x = self.beta * x + (1 - self.beta) * new_val
            self.data[key] = [new_x, new_step]

    def update_variable_avg(self,key,new_val):
        if key not in self.data.keys():
            self.add_variable(key, init_val=new_val)
            return
        x, step = self.data[key]
        new_step = step + 1
        new_x = (x * step + new_val) / new_step
        self.data[key] = [new_x, new_step]

    def get_variable_val(self, key):
        return self.data[key][0]

    def get_variable_str(self, key):
        return f'{key}: {self.data[key][0]:.5f}'

    def clear(self):
        self.data.clear()

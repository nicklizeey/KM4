import random
from itertools import product
import numpy as np

class ModTestdata:
    #Generates arithmetic elements based on the value of the mod
    def __init__(self, p):
        self.p = p
        self.data = list(product(set(range(p)), set(range(p))))
        random.shuffle(self.data)
        self.x, self.y = zip(*self.data)
    
    def x_list(self):
        return list(self.x)
    
    def y_list(self):
        return list(self.y)
    
# (x + y) % p
class AddModTransformerData:
    def __init__(self, x, y, fraction):
        self.x = x
        self.y = y
        self.fraction = fraction
        #The order is scrambled so that the model does not learn the order information of the data itself
        self.index2token = ['o', '='] + list(set(self.x).union(set(self.y)))
        random.shuffle(self.index2token)
        self.token2index = {v: i for i, v in enumerate(self.index2token)}
        # num_tokens is the number of unique tokens in the dataset, it also is p+2
        self.num_tokens = len(self.index2token)
        self.z = self.math_operation()
        #token of arithmetic operation
        self.data_token = []
        #Convert each mathematical symbol to a discrete symbol sequence number
        self.data_index = []
        self.constituent_seq()
        #Split the data into train and valid according to the fraction
        self.train, self.valid = self.split_data()

    def math_operation(self):
        z = (np.array(self.x) + np.array(self.y)) % (self.num_tokens - 2)
        return z.tolist()
    
    #use this method to map tokens to corresponding discrete serial numbers
    def encode(self, sequence):
        return [self.token2index[x] for x in sequence]

    #use this method to map the serial number to a token to verify the result
    def decode(self, sequence):
        return [self.index2token[x] for x in sequence]

    #Put x,y,z in a sequence
    def constituent_seq(self):
        for x, y, z in zip(self.x, self.y, self.z):
            tokens = [x, 'o', y, '=', z]
            indices = self.encode(tokens)
            self.data_token.append(tokens)
            self.data_index.append(indices)

    def split_data(self):
        split = int(len(self.data_index) * self.fraction)
        train = self.data_index[:split]
        valid = self.data_index[split:]
        return train, valid

# (x - y) % p 
class SubModTransformerData(AddModTransformerData):
    def math_operation(self):
        z = (np.array(self.x) - np.array(self.y)) % (self.num_tokens - 2)
        return z.tolist()
    
# (x * y) % p
class MulModTransformerData(AddModTransformerData):
    def math_operation(self):
        z = (np.array(self.x) * np.array(self.y)) % (self.num_tokens - 2)
        return z.tolist()
    
#When using division operations, you need to avoid zeros when generating data
#(x / y) % p
class DivModTransformerData(AddModTransformerData):
    def math_operation(self):
        z = (np.array(self.x) * pow(np.array(self.y), self.num_tokens - 4, self.num_tokens - 2)) % (self.num_tokens - 2)
        return z.tolist()
    
# (x^2 + y^2) % p
class Pow2ModTransformerData(AddModTransformerData):
    def math_operation(self):
        z = (np.array(self.x) ** 2 + np.array(self.y) ** 2) % (self.num_tokens - 2)
        return z.tolist()
    
# (x^3 + y^3) % p
class Pow3ModTransformerData(AddModTransformerData):
    def math_operation(self):
        z = (np.array(self.x) ** 3 + np.array(self.y) ** 3) % (self.num_tokens - 2)
        return z.tolist()
    
# (x^3 + xy) % p
class Powx3xyModTransformerData(AddModTransformerData):
    def math_operation(self):
        z = (np.array(self.x) ** 3 + np.array(self.x) * np.array(self.y)) % (self.num_tokens - 2)
        return z.tolist()

#(x^3 + xy^2 + y) % p
class Powx3xy2yModTransformerData(AddModTransformerData):
    def math_operation(self):
        z = (np.array(self.x) ** 3 + np.array(self.x) * np.array(self.y) ** 2 + np.array(self.y)) % (self.num_tokens - 2)
        return z.tolist()
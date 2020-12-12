import numpy as np
from numpy import random
from collections import Counter
import mmh3
import matplotlib.pyplot as plt
from scipy.stats import entropy
import math

# data generation parameter
stream_length = 1000

# initialize dictionaries
mg_gamma = {}
ss_gamma = {}
cmin_gamma = {}

# for i in range(1, 201, 20):
#     for j in range(1, 101, 10):
for i in (math.ceil(k) for k in np.logspace(0.0, 2, num=10)):
    for j in (math.ceil(l) for l in np.logspace(0.0, 2, num=10)):
        print('i & j:')
        print(i)
        print(j)
        data_sample_gamma = random.gamma(i, j, stream_length).astype(int)
        # fixed probability distribution (gamma distribution)

        # plt.hist(data_sample_gamma, density=True, bins=50)
        # plt.ylabel('Probability')
        # plt.xlabel('Data')
        # plt.show()

        # frequency count algorithms parameter
        num_counters = 100

        def get_frequencies(data):
            unique, counts = np.unique(data,return_counts=True)
            frequencies = list(zip(unique,counts))
            return(frequencies,len(unique))

        # calculate error for each items
        def get_errors(counters,frequencies):
            items = [t[0] for t in frequencies]
            counted = list()
            for key in list(counters.keys()):
                counted.append(key)
            uncounted = [item for item in items if item not in counted]
            errors = Counter()
            ## errors for items that don't have counters are their actual frequencies
            for item in uncounted:
                errors[item]+=[t[1] for t in frequencies if t[0] == item][0]
            for item in counted:
                errors[item]+=[t[1] for t in frequencies if t[0] == item][0]-counters[item]
            return(errors)

        def probabilities(frequencies):
            """Compute softmax values for each sets of scores in x."""
            probability = []
            sum = 0
            for i in range(frequencies[1]):
                sum = sum + frequencies[0][i][1]
            for i in range(frequencies[1]):
                probability.append(frequencies[0][i][1] / sum)
            return probability

        frequencies = get_frequencies(data_sample_gamma)
        print('actual frequencies:')
        print(frequencies)
        probability = probabilities(frequencies)
        print('actual probabilities:')
        print(probability)
        print('entropy:')
        ent = entropy(probability, base=2)
        print(entropy(probability, base=2))
        print('\n')

        def misra_gries(stream, k):
            counters = Counter()
            for item in stream:
                ## case 1: item already has counter or there are empty counters
                if item in counters or len(counters) < k:
                    counters[item] += 1
                ## case 2: item doesn't have counter and there are no empty counters
                else:
                    for key in list(counters.keys()):
                        counters[key] -= 1
                        if counters[key] == 0:
                            del counters[key]
            return counters

        # run MG on data
        out_mg = misra_gries(data_sample_gamma,int(num_counters))
        print('MG output:')
        print(out_mg)

        errors = get_errors(out_mg, frequencies[0])
        print('errors:')
        print(errors)

        total = 0
        N = math.ceil(float(frequencies[1])/10.0)
        for item in frequencies[0][:N]:
            total = total + abs(errors[item[0]])
        print('total error:')
        total = float(total) / float(N)
        print(total)
        print('\n')
        mg_gamma[ent] = total
        #plt.scatter(ent, total)
        #plt.show()

        def space_saving(stream, k):
            counters = Counter()
            for item in stream:
                ## case 1: item already has counter or there are empty counters
                if item in counters or len(counters) < k:
                    counters[item] += 1
                ## case 2: item doesn't have counter and there are no empty counters
                else:
                    min_key = min(counters, key = counters.get)
                    counters[min_key] += 1
                    counters[item] = counters.pop(min_key)
            return counters

        ## run space-saving on data
        #data = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 6], dtype = np.int32)
        out_ss = space_saving(data_sample_gamma,int(num_counters))
        print('SS output:')
        print(out_ss)

        errors = get_errors(out_ss, frequencies[0])
        print('errors:')
        print(errors)

        total = 0
        N = math.ceil(float(frequencies[1]) / 10.0)
        for item in frequencies[0][:N]:
            total = total + abs(errors[item[0]])
        print('total error:')
        total = float(total) / float(N)
        print(total)
        print('\n')

        ss_gamma[ent] = total
        # plt.scatter(ent, total)
        # plt.show()

        class CountMinSketch(object):
            ''' Class for a CountMinSketch data structure
            '''
            def __init__(self, width, depth, seeds):
                ''' Method to initialize the data structure
                @param width int: Width of the table
                @param depth int: Depth of the table (num of hash func)
                @param seeds list: Random seed list
                '''
                self.width = width
                self.depth = depth
                self.table = np.zeros([depth, width])  # Create empty table
                self.seed = seeds # np.random.randint(w, size = d) // create some seeds

            def increment(self, key):
                ''' Method to add a key to the CMS
                @param key str: A string to add to the CMS
                '''
                for i in range(0, self.depth):
                    index = mmh3.hash(key, self.seed[i]) % self.width
                    self.table[i, index] = self.table[i, index]+1

            def estimate(self, key, stream_len):
                ''' Method to estimate if a key is in a CMS
                @param key str: A string to check
                '''
                min_est = stream_len + 1
                for i in range(0, self.depth):
                    index = mmh3.hash(key, self.seed[i]) % self.width
                    if self.table[i, index] < min_est:
                        min_est = self.table[i, index]
                return min_est

        ## run space-saving on data
        #data = np.array([1, 2, 1, 1, 3, 2, 1, 1, 2, 3, 4, 2], dtype = np.int32)
        param_w = 100
        param_d = 10
        seeds = np.random.randint(param_w, size = param_d)
        cminsketch = CountMinSketch(param_w, param_d, seeds = seeds)
        for item in data_sample_gamma:
          cminsketch.increment(str(item))

        estimate_dict = {}
        for item in data_sample_gamma:
          if item not in estimate_dict:
            estimate_dict[item] =  cminsketch.estimate(str(item), stream_length)

        print('CMin output:')
        print(estimate_dict)

        errors = get_errors(estimate_dict, frequencies[0])
        print('errors:')
        print(errors)

        total = 0
        N = math.ceil(float(frequencies[1]) / 10.0)
        for item in frequencies[0][:N]:
            total = total + abs(errors[item[0]])
        print('total error:')
        total = float(total)/float(N)
        print(total)

        cmin_gamma[ent] = total
        # plt.scatter(ent, total)
        # plt.show()

x = []
y = []
for i in mg_gamma.keys():
    x.append(i)
    y.append(mg_gamma[i])

plt.scatter(x, y, s=4)
plt.title('Error vs Entropy for Misra-Gries on Gamma Distribution')
plt.xlabel('Entropy')
plt.ylabel('Error')
plt.show()

x = []
y = []
for i in ss_gamma.keys():
    x.append(i)
    y.append(ss_gamma[i])

plt.scatter(x, y, s=4)
plt.title('Error vs Entropy for Space Saving on Gamma Distribution')
plt.xlabel('Entropy')
plt.ylabel('Error')
plt.show()

x = []
y = []
for i in cmin_gamma.keys():
    x.append(i)
    y.append(cmin_gamma[i])

plt.scatter(x, y, s=4)
plt.title('Error vs Entropy for Count Min Sketch on Gamma Distribution')
plt.xlabel('Entropy')
plt.ylabel('Error')
plt.show()

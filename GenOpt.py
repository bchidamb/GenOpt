#===============================================================================
#   << GenOpt >>
#
# An automatic hyperparamater tuner based on genetic algorithm.
# Requires Python 3 (no dependencies)
#
# Created by Bhairav Chidambaram
#
# TODO:
#   - GPU Parallelization
#   - Detailed reports (including runtime) and .txt logs
#
#===============================================================================
# References
#-------------------------------------------------------------------------------
# DeepMind: Population Based Training of Neural Networks
# https://arxiv.org/pdf/1711.09846.pdf
#-------------------------------------------------------------------------------

import math
import random
import multiprocessing

# metaparameters
width = 10 # constant that affects perturbation in continuous and discrete variables
           # larger values take more generations to converge but are more precise
stay = 0.5 # probability of having the same value as parent when perturbing a categorical variable
keep = 1/5 # fraction of population used as parents for next generation

class Continuous:

    def __init__(self, low, high):
        '''
        Initializes a discrete variable.
        
        Arguments:
            low - the lowest decimal value this variable can take
            high - the highest decimal value this variable can take
        '''
        assert(isinstance(low, (int, float)))
        assert(isinstance(high, (int, float)))
        assert(0 < low and low <= high)
    
        self.min = low
        self.max = high
        self.delta = (high - low) ** (1 / width) - 1
        
    def sample(self):
        return math.exp(random.uniform(math.log(self.min), math.log(self.max)))
        
    def perturb(self, x):
        y = math.inf
        while self.min >= y or y >= self.max:
            y = x * (1 + random.uniform(-self.delta, self.delta))
    
        return y
        
    
class Discrete:

    def __init__(self, low, high):
        '''
        Initializes a discrete variable.
        
        Arguments:
            low - the lowest integer value this variable can take
            high - the highest integer value this variable can take
        '''
        assert(isinstance(low, int))
        assert(isinstance(high, int))
        assert(0 <= low and low <= high)
        
        self.min = low
        self.max = high
        self.delta = (high - low) ** (1 / width) - 1
        
    def sample(self):
        return math.floor(math.exp(random.uniform(math.log(self.min), math.log(self.max))))
        
    def perturb(self, x):
        y = math.inf
        while self.min > y or y > self.max:
            y = round(x * (1 + random.uniform(-self.delta, self.delta)))
        
        return y
        
        
class Categorical:

    def __init__(self, values):
        '''
        Initializes a categorical variable.
        
        Arguments:
            values - a list of objects
        '''
        assert(isinstance(values, list))
    
        self.values = values
        
    def sample(self):
        return random.choice(self.values)
    
    def perturb(self, x):
        return x if random.random() < stay else self.sample()
        

class _ParameterSet:
    # internal class that holds a parameter assignment to all variables
    
    def __init__(self, vars, parent=None):
        if parent:
            params = []
            # asexual reproduction
            for i, v in enumerate(vars):
                params.append(v.perturb(parent.params[i]))
            self.params = tuple(params)
        else:
            self.params = tuple(v.sample() for v in vars)
        self.loss = None
        
    def __lt__(self, other):
        if self.loss is None or other.loss is None:
            raise Exception('Bad ParameterSet comparison')
        return self.loss < other.loss
        
    def __str__(self):
        return self.loss

        
def _evaluate(eval, population, threads):
    # internal function for evaluating population in parallel
    
    pop_params = [p.params for p in population]
    results = []
    
    if threads == 1:
        results = map(eval, pop_params)
        
    else:
        # https://stackoverflow.com/questions/884650
        pool = multiprocessing.Pool(threads)
        a = pool.map_async(eval, pop_params, callback=results.append)
        a.wait()
    
    for i, r in enumerate(results):
        population[i].loss = r
    

def optimize(eval, vars, pop=10, gen=10, threads=1):
    '''
    Use this function to optimize your model's hyperparamaters.
    
    Arguments:
        eval - a function which outputs validation loss given parameter tuple as input
        pop - the number of members in each generation
        gen - the number of generations that genetic algorithm will run
        threads - the number of threads to use in parallel computation of eval
    '''
    for g in range(gen+1):
    
        if g == 0:
            population = [_ParameterSet(vars) for _ in range(pop)]
            
        else:
            population.sort()
            parents = population[: int(pop * keep)]
            population = [_ParameterSet(vars, parent=random.choice(parents)) for _ in range(pop)]
        
        _evaluate(eval, population, threads)
        best = min(population)
        
        print('Generation %d' % g)
        print('Best score: %.4f' % best.loss)
            
    print('Best parameter set:', best.params)
    
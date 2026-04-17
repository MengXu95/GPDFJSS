import copy
import random
import numpy as np

from deap import gp, creator
from deap import tools
# from MTGP.GPFC import N_TREES, MAX_HEIGHT


# def process_data(individual, toolbox, data):
#     no_instances = data.shape[0]
#     no_trees = len(individual)
#     feature_major = data.T
#     # [no_trees x no_instances]
#     # we do it this way so we can assign rows (constructed features) efficiently.
#     result = np.zeros(shape=(no_trees, no_instances))
#     for i, expr in enumerate(individual):
#         func = toolbox.compile(expr=expr)
#         vec = func(*feature_major)
#         if (not isinstance(vec, np.ndarray)) or vec.ndim == 0:
#             # it decided to just give us a constant back...
#             vec = np.repeat(vec, no_instances)
#         result[i] = vec
#     return result.T


def init_primitives_tree1(pset):
    # add function
    # pset.addPrimitive(np.add, 2)
    # pset.addPrimitive(np.subtract, 2)
    pset.addPrimitive(np.multiply, 2)
    pset.addPrimitive(protected_div, 2)
    pset.addPrimitive(np.maximum, 2)
    pset.addPrimitive(np.minimum, 2)
    # pset.addPrimitive(lf, 1)  # add by mengxu 2022.11.08 for GSGP
    # pset.addPrimitive(add_abs, 2)
    # pset.addPrimitive(sub_abs, 2)
    #pset.addPrimitive(mt_if, 3)
    #pset.addEphemeralConstant("rand", ephemeral=lambda: random.uniform(-1, 1))
    # add terminal
    # pset.addTerminal(1)  # add by mengxu //todo: the terminals seems not right, it already has three terminals in the set before I add my terminals in, need to modify
    # pset.addTerminal(2)  # add by mengxu
    # pset.addTerminal(3)  # add by mengxu

    # terminals for sequencing and routing in my paper
    # pset.addTerminal(str('NIQ'))  # add by mengxu
    # pset.addTerminal(str('WIQ'))  # add by mengxu
    pset.addTerminal(str('MWT'))  # add by mengxu
    pset.addTerminal(str('PT'))  # add by mengxu
    pset.addTerminal(str('NPT'))  # add by mengxu
    pset.addTerminal(str('OWT'))  # add by mengxu
    pset.addTerminal(str('WKR'))  # add by mengxu
    pset.addTerminal(str('NOR'))  # add by mengxu
    # pset.addTerminal('W')  # add by mengxu
    pset.addTerminal(str('TIS'))  # add by mengxu
    # pset.addTerminal('TRANT')  # add by mengxu

    # adviced terminal by Yi 2022.10.31
    pset.addTerminal(str('SLACK'))  # add by mengxu

    # pset.addTerminal('NIQ')  # add by mengxu
    # pset.addTerminal('WIQ')  # add by mengxu
    # pset.addTerminal('MWT')  # add by mengxu
    # pset.addTerminal('PT')  # add by mengxu
    # pset.addTerminal('NPT')  # add by mengxu
    # pset.addTerminal('OWT')  # add by mengxu
    # pset.addTerminal('WKR')  # add by mengxu
    # pset.addTerminal('NOR')  # add by mengxu
    # # pset.addTerminal('W')  # add by mengxu
    # pset.addTerminal('TIS')  # add by mengxu
    # # pset.addTerminal('TRANT')  # add by mengxu


    # # terminals for sequencing
    # pset.addTerminal('current_pt') #add by mengxu //todo: the terminals seems not right, it already has three terminals in the set before I add my terminals in, need to modify
    # pset.addTerminal('slack') #add by mengxu
    # pset.addTerminal('queue') #add by mengxu
    #
    # # terminals for routing
    # pset.addTerminal('time_in_system')  # add by mengxu
    # pset.addTerminal('que_size')  # add by mengxu

def init_primitives_tree2(pset):
    # add function
    pset.addPrimitive(np.add, 2)
    pset.addPrimitive(np.subtract, 2)

    # terminals for sequencing and routing in my paper
    pset.addTerminal(str('NIQ'))  # add by mengxu
    pset.addTerminal(str('WIQ'))  # add by mengxu



def lf(x): # add by mengxu 2022.11.08
    return 1 / (1 + np.exp(-x))


def init_toolbox(toolbox, pset1, pset2):
    creator.create("Individual", list, fitness=creator.FitnessMin, pset1=pset1, pset2=pset2)

    #original: if we have two tree and they use the same terminal set
    # toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=6) # original max = 6, modified by mengxu 2022.10.15 to check
    # toolbox.register("tree", tools.initIterate, gp.PrimitiveTree, toolbox.expr)
    # N_TREES = 2 #todo: only for test, need to be the same with original GPFC.py
    # toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.tree, n=N_TREES)

    #modified: if we have two tree and they use different terminal sets by mengxu
    toolbox.register("expr1", gp.genHalfAndHalf, pset=pset1, min_=1,
                     max_=6)  # original max = 6, modified by mengxu 2022.10.15 to check
    toolbox.register("tree1", tools.initIterate, gp.PrimitiveTree, toolbox.expr1)
    toolbox.register("expr2", gp.genHalfAndHalf, pset=pset2, min_=1,
                     max_=6)  # original max = 6, modified by mengxu 2022.10.15 to check
    toolbox.register("tree2", tools.initIterate, gp.PrimitiveTree, toolbox.expr2)
    N_TREES = 1  # todo: only for test, need to be the same with original GPFC.py
    toolbox.register("individual", tools.initCycle, creator.Individual, [toolbox.tree1, toolbox.tree2], n=N_TREES)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset1)
    toolbox.register("compile", gp.compile, pset=pset2)

    toolbox.register("expr_mut", gp.genFull, min_=2, max_=8)
    #toolbox.register("mate", xmate)
    #toolbox.register("mutate", xmut, expr=toolbox.expr_mut)

    toolbox.register("mate",lim_xmate)
    toolbox.register("mutate",lim_xmut,expr=toolbox.expr_mut)



def maxheight(v):
    return max(i.height for i in v)


# stolen from gp.py....because you can't pickle decorated functions.
def wrap(func, *args, **kwargs):
    MAX_HEIGHT = 8 #todo: only for test, need to be the same with original GPFC.py
    keep_inds = [copy.deepcopy(ind) for ind in args]
    new_inds = list(func(*args, **kwargs))
    for i, ind in enumerate(new_inds):
        if maxheight(ind) > MAX_HEIGHT:
            new_inds[i] = random.choice(keep_inds)
    return new_inds

# the following is modified by mengxu
def xmate(ind1, ind2):
    i1 = random.randrange(len(ind1))
    # i2 = random.randrange(len(ind2))
    #todo: I think this is not same with my MTGP, as only the same type of tree can be used to do crossover
    ind1[i1], ind2[i1] = gp.cxOnePoint(ind1[i1], ind2[i1])

    #exchange the other tree
    i2 = 1 - i1 # only for individual with two tree
    ind1[i2], ind2[i2] = ind2[i2], ind1[i2]
    return ind1, ind2

# def xmate(ind1, ind2):
#     i1 = random.randrange(len(ind1))
#     i2 = random.randrange(len(ind2))
#     ind1[i1], ind2[i2] = gp.cxOnePoint(ind1[i1], ind2[i2])
#     return ind1, ind2


def lim_xmate(ind1, ind2):
    return wrap(xmate, ind1, ind2)


def xmut(ind, expr):
    i1 = random.randrange(len(ind))
    indx = gp.mutUniform(ind[i1], expr,pset=ind.pset)
    ind[i1] = indx[0]
    return ind,


def lim_xmut(ind, expr):
    # have to put expr=expr otherwise it tries to use it as an individual
    res = wrap(xmut, ind, expr=expr)
    # print(res)
    return res


def add_abs(a, b):
    return np.abs(np.add(a, b))


def sub_abs(a, b):
    return np.abs(np.subtract(a, b))


def mt_if(a, b, c):
    return np.where(a < 0, b, c)


def protected_div(left, right):
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x

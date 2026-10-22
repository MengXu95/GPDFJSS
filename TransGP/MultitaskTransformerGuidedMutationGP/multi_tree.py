import random
import numpy as np

from deap import gp, creator
from deap import tools
from MTGPforNewSimulatorBasedOnJava.multi_tree import init_primitives, lim_xmate, wrap

task_embedding_dict = {
        "task_0": [0.75, 6],
        "task_1": [0.85, 8],
        "task_2": [0.95, 10]
    }

def init_toolbox(toolbox, pset):
    creator.create("Individual", list, fitness=creator.FitnessMin, pset=pset, task_tag=0)

    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=5) # original max = 6, modified by mengxu 2022.10.15 to check
    toolbox.register("tree", tools.initIterate, gp.PrimitiveTree, toolbox.expr)
    N_TREES = 2 #todo: only for test, need to be the same with original GPFC.py
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.tree, n=N_TREES)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    # toolbox.register("expr_mut", gp.genGrow, min_=1, max_=6)
    toolbox.register("expr_mut", gp.genFull, min_=1, max_=3) # normally used, original one
    #toolbox.register("mate", xmate)
    #toolbox.register("mutate", xmut, expr=toolbox.expr_mut)

    toolbox.register("mate",lim_xmate)
    print("Use transformer model guided mutation!")
    toolbox.register("mutate", lim_xmut_transformer, expr=toolbox.expr_mut) # A pre-trained diffusion model guided mutation
    # toolbox.register("mutate", lim_xmut, expr=toolbox.expr_mut) # original



def xmut(ind, expr, transformer_gp):
    task_num = len(task_embedding_dict.values())
    i1 = random.randrange(len(ind))
    # mutate the task_tag if random value < mutate_pro
    if np.random.rand() < 0.2:
        rand_task_tag = np.random.randint(task_num)
        ind.task_tag = rand_task_tag
    indx = gp.mutUniform(ind[i1], expr, pset=ind.pset)
    ind[i1] = indx[0]
    return ind,

def lim_xmut(ind, expr, transformer_gp=None):
    # have to put expr=expr otherwise it tries to use it as an individual
    res = wrap(xmut, ind, expr=expr, transformer_gp=transformer_gp) # original
    # print(res)
    return res


def xmut_transformer_guided(ind, expr, transformer_gp):
    task_num = len(task_embedding_dict.values())
    i1 = random.randrange(len(ind))
    # mutate the task_tag if random value < mutate_pro
    if np.random.rand() < 0.2:
        rand_task_tag = np.random.randint(task_num)
        ind.task_tag = rand_task_tag
    dict_name = "task_"+str(ind.task_tag)
    task_embedding = task_embedding_dict.get(dict_name)
    indx = transformer_gp[i1].mutate_tree(ind[i1], task_embedding, pset=ind.pset)
    ind[i1] = indx[0]
    return ind,

# def xmut_diffusion_guided(ind, expr, diffusion_gp):
#     for i in range(len(ind)):
#         indx = diffusion_gp[i].diffusion_guided_mutation(ind[i], pset=ind.pset)
#         ind[i] = indx[0]
#     return ind,

def lim_xmut_transformer(ind, expr, transformer_gp=None):
    # have to put expr=expr otherwise it tries to use it as an individual
    res = wrap(xmut_transformer_guided, ind, expr=expr, transformer_gp=transformer_gp)
    # print(res)
    return res


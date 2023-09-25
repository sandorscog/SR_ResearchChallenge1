from time import time

import numpy as np
import pandas as pd


def cosine(u, v):

    dot_product = np.sum(np.dot(u, v))
    u_sqrt = np.sqrt(np.sum(np.square(u)))
    v_sqrt = np.sqrt(np.sum(np.square(v)))

    cos = dot_product / (u_sqrt * v_sqrt)
    return cos


def neighborhood_viability(object_index):

    not_viable = []
    for obj in object_index.keys():
        raters = object_index[obj]
        if len(raters) < 2:
            not_viable.append(obj)

    return not_viable


def neighbors(object_index, user_index, target_items):

    neighborhoods = {key: [] for key in target_items}

    for item in neighborhood_viability(object_index):
        if item in target_items:
            target_items.remove(item)
            neighborhoods[item].append(('x', 0))

    tempting_objects = []
    for i in target_items:
        users = list(dict(object_index[i]).keys())

        for u in users:
            item_names = list(dict(user_index[u]).keys())
            tempting_objects.extend(item_names)

    tempting_objects = list(set(tempting_objects))

    # # # # # # # # # # # # # # #
    for i in target_items:
        for j in tempting_objects[:150]:
            u = []
            v = []
            if i == j:
                continue

            for ui in object_index[i]:
                for vi in object_index[j]:
                    if ui[0] == vi[0]:
                        u.append(ui[1])
                        v.append(vi[1])

            if len(u) == 0:
                cos = 1
            elif all(i == 0 for i in u) or all(i == 0 for i in v):
                cos = 1
            else:
                cos = cosine(u, v)

            neighborhoods[i].append((j, cos))

    # print(neighborhoods)
    return neighborhoods

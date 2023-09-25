import sys
from random import randint
from time import time

import numpy as np
import pandas as pd


PATH = 'C:/Users/Desktop/Desktop/Misc/UFMG/Isolada/23_2/SistemasDeRocomendacao/RC1/'


def users_means(ratings):
    scores = {key: [] for key in ratings['UserId'].unique().tolist()}

    for row in ratings.itertuples():
        scores[row.UserId].append(row.Rating)

    for key in scores.keys():
        values = scores[key]
        mean = np.sum(values)/len(values)
        scores[key] = mean

    return scores


def items_means(ratings):
    scores = {key: [] for key in ratings['ItemId'].unique().tolist()}

    for row in ratings.itertuples():
        scores[row.ItemId].append(row.Rating)

    for key in scores.keys():
        values = scores[key]
        mean = np.sum(values)/len(values)
        scores[key] = mean

    return scores


def dense_representation_user(data, mean):

    dense_representation = {key: [] for key in data['UserId'].unique().tolist()}

    for row in data.itertuples():
        dense_representation[row.UserId].append((row.ItemId, row.Rating - mean[row.UserId]))

    return dense_representation


def dense_representation_builder(data, mean):

    dense_representation = {key: [] for key in data['ItemId'].unique().tolist()}

    for row in data.itertuples():
        dense_representation[row.ItemId].append((row.UserId, row.Rating-mean[row.UserId]))

    return dense_representation


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
        #print(item)
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
        for j in tempting_objects[:250]:
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


def predict_ratings(targets, neighborhood, user_ratings, mean_values, items_mean_values):

    predictions = pd.DataFrame({'UserId:ItemId': [], 'Rating': []})

    for target in targets.itertuples():

        target_tuple = str(target.UserId) + ':' + str(target.ItemId)

        if neighborhood[target.ItemId][0][0] == 'x' or True:
            final_rating = items_mean_values[target.ItemId]

        else:
            prediction = rating_prediction(target.ItemId, target.UserId, neighborhood, user_ratings)
            prediction += mean_values[target.UserId]

            try:
                final_rating = int(round(prediction, 0))
            except:
                final_rating = randint(1, 5)

        prediction = pd.DataFrame({'UserId:ItemId': [target_tuple], 'Rating': [final_rating]})
        predictions = pd.concat([predictions, prediction])

    return predictions


def rating_prediction(item_t, user_t, neighborhood, user_ratings):

    user_t_ratings = dict(user_ratings[user_t])

    numerator = 0
    # print(user_t_ratings)
    for neighbor in neighborhood[item_t]:
        if user_t_ratings.get(neighbor[0]) is not None and not np.nan:
            numerator += user_t_ratings[neighbor[0]]*neighbor[1]

    denominator = 0
    for neighbor in neighborhood[item_t]:
        if user_t_ratings.get(neighbor[0]) is not None and not np.nan:
            denominator += neighbor[1]

    if denominator == 0:
        denominator = .1

    predict = numerator/denominator
    return predict


def main():
    init = time()

    # Data read
    ratings = pd.read_csv(PATH + 'ratings.csv')
    targets = pd.read_csv(PATH + 'targets.csv')

    # Split association User:Item
    ratings[['UserId', 'ItemId']] = ratings['UserId:ItemId'].str.split(':', 1, expand=True)
    targets[['UserId', 'ItemId']] = targets['UserId:ItemId'].str.split(':', 1, expand=True)

    mean_values = users_means(ratings)
    item_means = items_means(ratings)
    item_dense_representation = dense_representation_builder(ratings, mean_values)
    user_dense_representation = dense_representation_user(ratings, mean_values)
    #print('Neighbors')
    neighborhood = neighbors(item_dense_representation, user_dense_representation, targets['ItemId'].unique().tolist())
    #print('Predictions')
    predictions = predict_ratings(targets, neighborhood, user_dense_representation, mean_values, item_means)

    predictions = predictions.astype({'Rating': int})
    predictions.to_csv(PATH+'resultado.csv', index=False)

    print(time()-init)


def teste():

    ratings = pd.read_csv(PATH+'ratings_teste.csv')
    targets = pd.DataFrame({'UserId': ['A', 'B', 'A'], 'ItemId': ['banana', 'laranja', 'tomate']})

    mean_values = users_means(ratings)
    item_means = items_means(ratings)

    dense_representation = dense_representation_builder(ratings, mean_values)
    user_dense_representation = dense_representation_user(ratings, mean_values)

    neighborhood = neighbors(dense_representation, user_dense_representation, targets['ItemId'].unique().tolist())
    predictions = predict_ratings(targets, neighborhood, user_dense_representation, mean_values, item_means)
    predictions = predictions.astype({'Rating': int})

    predictions.to_csv(PATH + 'resultado.csv', index=False)


if __name__ == '__main__':
    main()
    # teste()

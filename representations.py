from time import time

import numpy as np
import pandas as pd


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


def dense_representation_item(data, mean):

    dense_representation = {key: [] for key in data['ItemId'].unique().tolist()}

    for row in data.itertuples():
        dense_representation[row.ItemId].append((row.UserId, row.Rating-mean[row.UserId]))

    return dense_representation

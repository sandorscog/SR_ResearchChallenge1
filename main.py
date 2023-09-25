import sys
from time import time

import numpy as np
import pandas as pd

from neighborhood import neighbors
from representations import *
from prediction import predict_ratings


def main(ratings_file, targets_file):
    init = time()

    # Data read
    ratings = pd.read_csv(ratings_file)
    targets = pd.read_csv(targets_file)

    # Split association User:Item
    ratings[['UserId', 'ItemId']] = ratings['UserId:ItemId'].str.split(':', 1, expand=True)
    targets[['UserId', 'ItemId']] = targets['UserId:ItemId'].str.split(':', 1, expand=True)

    # Find the mean values for the rows and columns
    mean_values = users_means(ratings)
    item_means = items_means(ratings)

    # Convert the ratings to a dense representation in Dict format
    item_dense_representation = dense_representation_item(ratings, mean_values)
    user_dense_representation = dense_representation_user(ratings, mean_values)

    # Define the relationship between items aka their neighbors
    neighborhood = neighbors(item_dense_representation, user_dense_representation, targets['ItemId'].unique().tolist())

    # Predict all the ratings for each target
    predictions = predict_ratings(targets, neighborhood, user_dense_representation, mean_values, item_means)

    # Write the results to a csv File
    predictions = predictions.astype({'Rating': float})
    predictions.to_csv('result.csv', index=False)

    print('Execution time:', time()-init, 'seconds')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

import numpy as np
import pandas as pd


def rating_prediction(item_t, user_t, neighborhood, user_ratings):

    user_t_ratings = dict(user_ratings[user_t])

    numerator = 0
    for neighbor in neighborhood[item_t]:
        if user_t_ratings.get(neighbor[0]) is not None and not np.nan:
            numerator += user_t_ratings[neighbor[0]]*neighbor[1]

    denominator = 0
    for neighbor in neighborhood[item_t]:
        if user_t_ratings.get(neighbor[0]) is not None and not np.nan:
            denominator += neighbor[1]

    # Safety for division by 0
    if denominator == 0:
        denominator = .1

    predict = numerator/denominator
    return predict


def predict_ratings(targets, neighborhood, user_ratings, mean_values, items_mean_values):

    predictions = pd.DataFrame({'UserId:ItemId': [], 'Rating': []})

    for target in targets.itertuples():

        target_tuple = str(target.UserId) + ':' + str(target.ItemId)

        # Checks if the item was given neighbors, returns the item average if not
        if neighborhood[target.ItemId][0][0] == 'x':
            final_rating = items_mean_values[target.ItemId]

        else:
            # Predicts the rating value and aggregates it to the item mean
            prediction_value = rating_prediction(target.ItemId, target.UserId, neighborhood, user_ratings)
            prediction_value += (mean_values[target.UserId] + items_mean_values[target.ItemId])/2

            final_rating = prediction_value

        # Print the result to the std output
        print(target_tuple, final_rating)

        prediction = pd.DataFrame({'UserId:ItemId': [target_tuple], 'Rating': [final_rating]})
        predictions = pd.concat([predictions, prediction])

    return predictions

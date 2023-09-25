import sys

import numpy as np
import pandas as pd


PATH = 'C:/Users/Desktop/Desktop/Misc/UFMG/Isolada/23_2/SistemasDeRocomendacao/RC1/'


def main():

    # Data read
    ratings = pd.read_csv(PATH + 'ratings.csv')
    targets = pd.read_csv(PATH + 'targets.csv')

    # Split association User:Item
    targets[['UserId', 'ItemId']] = ratings['UserId:ItemId'].str.split(':', 1, expand=True)
    ratings[['UserId', 'ItemId']] = ratings['UserId:ItemId'].str.split(':', 1, expand=True)

    target_users = set(targets['UserId'].unique().tolist())
    target_items = set(targets['ItemId'].unique().tolist())

    ratings = ratings.query('UserId in @target_users and ItemId in @target_items')
    # print(ratings['ItemId'].value_counts(), ratings['ItemId'].value_counts().index.tolist()[:5])
    item_order = ratings['ItemId'].value_counts().index.tolist()
    best_items = ratings['ItemId'].value_counts().index.tolist()[:5]


    # corte = 1
    # user_ids = {key: val for key, val in user_ids.items() if val > corte}
    # item_ids = {key: val for key, val in item_ids.items() if val > corte}
    print(len(set(ratings['ItemId'].unique().tolist())))
    print(len(set(ratings['UserId'].unique().tolist())))

    relevant_ratings = pd.DataFrame({}, columns=target_items, index=target_users)
    relevant_ratings = relevant_ratings.reindex(columns=item_order)

    # relevant_ratings_transposed = relevant_ratings.transpose()
    for index, row in ratings.iterrows():
        relevant_ratings.at[row['UserId'], row['ItemId']] = row['Rating']

    print(relevant_ratings[best_items[0]].value_counts())
    # print(relevant_ratings.count(axis='columns'))

    # print(relevant_ratings.columns)
    # print(relevant_ratings['8b05db84f2'].dropna())

    items = np.asarray(list(target_items))
    # print(target_items)
    print('Ponto')
    #M = np.asarray(relevant_ratings.loc[:, items])
    #M_u = M.mean(axis=1)
    #item_mean_subtracted = M - M_u[:, None]

    M = np.asarray(relevant_ratings.loc[:, items])
    M_u = np.nanmean(M, axis=1)
    item_mean_subtracted = M.transpose() - M_u[:]
    ratings = item_mean_subtracted.transpose()

    print(M)
    print(type(M))
    print(ratings)

    print('Vizinhos')
    #np.set_printoptions(threshold=np.inf)
    print(ratings[:, 0])
    #np.set_printoptions(threshold=5)
    print(neighbors_correlation(ratings, neighborhood_size=3))
    '''
    '''

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def cosine(u, v):
    # dot_product = 0
    # for i, j in zip(u, v):
    #     dot_product += i*j


    dot_product = np.nansum(u*v)
    u_sqrt = np.sqrt(np.nansum(np.square(u)))
    v_sqrt = np.sqrt(np.nansum(np.square(v)))

    cos = dot_product/(u_sqrt*v_sqrt)
    return cos


def neighbors_correlation(ratings, neighborhood_size=3):

    neighbors = {key: [] for key in range(ratings.shape[1])}

    for i in range(neighborhood_size):
        for j in range(2, ratings.shape[1]):
            cos = cosine(ratings[:, i], ratings[:, j])
            # neighbors[i].append(cos)
            neighbors[j].append(cos)

    neighborhood = range(neighborhood_size+1)
    for i in neighborhood:
        adj_neighborhood = list(neighborhood)
        adj_neighborhood.remove(i)
        for j in adj_neighborhood:
            cos = cosine(ratings[:, i], ratings[:, j])
            neighbors[i].append(cos)

    # neighbors = sort_neighbors(neighbors)
    return neighbors


def sort_neighbors(neighbors):

    sorted_neighbors = {key: [] for key in neighbors.keys()}
    for i in neighbors.keys():

        neighbor_list = neighbors[i]
        if len(neighbor_list) > 3:

            for iteration in range(3):
                n = max(neighbor_list)
                sorted_neighbors[i].append(n)
                neighbor_list.remove(n)

        else:
            sorted_neighbors[i].extend(neighbor_list)

    return sorted_neighbors


def teste_cosine():

    ratings = pd.read_csv(PATH+'ratings_teste.csv')
    print(ratings)
    print(ratings.reindex(columns=ratings.columns.sort_values()))

    items = np.asarray(list(ratings.columns))
    print(items)
    M = np.asarray(ratings.loc[:, items])
    M_u = np.nanmean(M, axis=1)
    item_mean_subtracted = M.transpose() - M_u[:]
    ratings = item_mean_subtracted.transpose()

    print(M)
    print(type(M))
    print(M_u)
    print(ratings)
    print(ratings[:, 0])

    print(ratings.shape[1])

    # u = np.array([4, 2, 3])
    # v = np.array([-34, 26, 2])

    # print(np.nansum(u*v))
    print(neighbors_correlation(ratings))


if __name__ == '__main__':
    #pd.set_option('display.min_rows', 1000)

    print(sys.argv)  # lista com os argumentos do prompt ['main.py', 'mariana']

    #teste_cosine()
    main()

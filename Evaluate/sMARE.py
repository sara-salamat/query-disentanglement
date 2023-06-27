import numpy as np


def calculate_sare(qpprank, actualrank, Q):
    '''
    qpprank: rank of a single query in ranked list with a qpp approach
    actualrank: actual rank of a single query in a ranked list
    Q: number of queries in the ranked list
    '''
    return np.abs(qpprank-actualrank)/Q

def calculate_sMARE(list_of_qppranks, list_of_actualranks):
    Q = len(list_of_qppranks)
    sARE_scores = 0
    for i in range(Q):
        sARE_scores = sARE_scores + calculate_sare(list_of_qppranks[i], list_of_actualranks[i], Q)
    return sARE_scores/Q
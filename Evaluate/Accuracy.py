import pathlib

import pandas as pd
import random
import io
import torch
from scipy.stats import kendalltau, spearmanr, pearsonr
from transformers import AutoTokenizer
from sMARE import calculate_sMARE


path_to_data = pathlib.Path('data')
path_to_model = pathlib.Path('qpp-mpnet-classification-2023-06-15_00-02-43')
# path_to_model = pathlib.Path('output/qpp-bert-reverse-classification-2023-05-29_21-39-34')

#### loading models
# with open(path_to_model / 'model_ep1.pkl', 'rb') as f:
#     pred_model = torch.load(f)
q_19 = pd.read_csv(path_to_data / 'msmarco-test2019-queries.tsv', sep='\t', header=None) ## qid \t query
dl_19 = pd.read_csv(path_to_data / 'dl2019_ndcg10', sep='\t', header=None) ## qid \t map
print('data read')
queries_numbers = []
maps = []
# for i in range(len(dl_19)):
#     # if dl_19.iloc[i][1].isdigit():
#     queries_numbers.append(dl_19.iloc[i][1])
#     maps.append((dl_19.iloc[i][1], dl_19.iloc[i][2]))
# selected_queries = q_19[q_19[0].isin(list(map(int, queries_numbers)))] ## finding the queries

for i in range(len(dl_19)):
    # if dl_19.iloc[i][1].isdigit():
    queries_numbers.append(dl_19.iloc[i][1])
    maps.append((dl_19.iloc[i][1], dl_19.iloc[i][2]))
selected_queries = q_19[q_19[0].isin(list(map(int, queries_numbers)))] ## finding the queries

with open(path_to_model / 'model_ep4.pkl', 'rb') as f:
    pred_model = torch.load(f)


Task = 'classification'
model_name = 'microsoft/deberta-v3-base'
# model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)

def to_cuda(Tokenizer_output):
    tokens_tensor = Tokenizer_output['input_ids'].to('cuda')
    token_type_ids = Tokenizer_output['token_type_ids'].to('cuda')
    attention_mask = Tokenizer_output['attention_mask'].to('cuda')
    output = {'input_ids' : tokens_tensor, 
              'token_type_ids' : token_type_ids, 
              'attention_mask' : attention_mask}
    return output



def is_lower(a,b):
    q_a = selected_queries[selected_queries[0]==a][1].values[0]
    q_b = selected_queries[selected_queries[0]==b][1].values[0]

    
    predicted_score = pred_model.predict(
    to_cuda(tokenizer(q_a.lower(),return_tensors='pt')),
    to_cuda(tokenizer(q_b.lower(),return_tensors='pt')),
    )
    predicted_score_v = pred_model.predict(
    to_cuda(tokenizer(q_b.lower(),return_tensors='pt')),
    to_cuda(tokenizer(q_a.lower(),return_tensors='pt')),
    )

    predicted_difference_score = 0.5*(predicted_score + 1 - predicted_score_v)# Formula
    # predicted_difference_score = predicted_score
    if Task == 'regression':
        threshold = 0
    elif Task == 'classification':
        threshold = 0.5
    else: 
        print('Wrong task?')
    if  predicted_difference_score.item()<threshold:
        return True
    else:
        return False

# maps_dict = dict(maps)
# def true_label(a,b):
#     map_a = maps_dict[str(a)]
#     map_b = maps_dict[str(b)]
#     if map_a<map_b:
#         return True
#     else:
#         return False

maps_dict = dict(maps)
def true_label(a,b):
    map_a = maps_dict[a]
    map_b = maps_dict[b]
    if map_a<map_b:
        return True
    else:
        return False

def _insort_right(a, x, q, counter, acc_counter):
    """
    Insert item x in list a, and keep it sorted assuming a is sorted.
    If x is already in a, insert it to the right of the rightmost x.
    """
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo+hi)//2
        q += 1
        less = is_lower(x, a[mid])
        
        accurate = true_label(x, a[mid])
        counter = counter +1

        if accurate==less:
            acc_counter = acc_counter+1
        if less: hi = mid
        else: lo = mid+1
    a.insert(lo, x)
    return q , counter, acc_counter

def order(items):
    ordered, q = [], 0
    counter = 0
    acc_counter = 0
    for item in items:
        q , counter, acc_counter = _insort_right(ordered, item, q, counter, acc_counter)
    print('accuracy: ', acc_counter/counter)
    print('total comparisons: ', counter)
    return ordered, q

a = 'How old is obama?'
b = 'Explain ebola virus and its effects on human life'
print('------ a-b b-a ------')
print(pred_model.predict(to_cuda(tokenizer(a.lower(),return_tensors='pt')), to_cuda(tokenizer(b.lower(),return_tensors='pt'))))

print(pred_model.predict(to_cuda(tokenizer(b.lower(),return_tensors='pt')), to_cuda(tokenizer(a.lower(),return_tensors='pt'))))

print('===== 1 =====')
ordered_list, q = order(selected_queries[0].to_list())
sorted_q19 = list(sorted(maps, key=lambda x: x[1], reverse=False))
actual_sorting = list(map(lambda x: int(x[0]),sorted_q19))
predicted_sorting = ordered_list
list_1 = []
list_2 = []
for qid in selected_queries[0]:
    list_1.append(actual_sorting.index(qid))
    list_2.append(predicted_sorting.index(qid))


print(kendalltau(list_1,list_2))
print(spearmanr(list_1, list_2))
print(pearsonr(list_1, list_2))
print('smare: ', calculate_sMARE(list_1, list_2))

# print('===== 2 =====')
# to_order = selected_queries[0].to_list()
# random.shuffle(to_order)
# ordered_list, q = order(to_order)
# predicted_sorting = ordered_list
# list_2 = []
# for qid in selected_queries[0]:
#     list_2.append(predicted_sorting.index(qid))

# print(kendalltau(list_1,list_2))
# print(spearmanr(list_1, list_2))

# print('===== 3 =====')
# to_order = selected_queries[0].to_list()
# random.shuffle(to_order)
# ordered_list, q = order(to_order)
# predicted_sorting = ordered_list
# list_2 = []
# for qid in selected_queries[0]:
#     list_2.append(predicted_sorting.index(qid))

# print(kendalltau(list_1,list_2))
# print(spearmanr(list_1, list_2))

# print('===== 4 =====')
# to_order = selected_queries[0].to_list()
# random.shuffle(to_order)
# ordered_list, q = order(to_order)
# predicted_sorting = ordered_list
# list_2 = []
# for qid in selected_queries[0]:
#     list_2.append(predicted_sorting.index(qid))

# print(kendalltau(list_1,list_2))
# print(spearmanr(list_1, list_2))
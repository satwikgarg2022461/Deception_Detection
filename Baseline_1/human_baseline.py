import json
from sklearn.metrics import f1_score, accuracy_score
from random import uniform
import jsonlines

test_set = 'dataset/test.jsonl'

total_tt, total_tn, total_nt, total_nn = 0, 0, 0, 0
sender_labels = []
receiver_labels = []


def aggregate(dataset):
    messages = []
    rec = []
    send = []
    for dialogs in dataset:
        messages.extend(dialogs['messages'])
        rec.extend(dialogs['receiver_labels'])
        send.extend(dialogs['sender_labels'])
    merged = []
    for i, item in enumerate(messages):
        merged.append({'message':item, 'sender_annotation':send[i], 'receiver_annotation':rec[i]})
    return merged

def process_message(msg):
    global total_tt, total_tn, total_nt, total_nn
    
    if msg['receiver_annotation'] != True and msg['receiver_annotation'] != False:
        return None, None
    
    sender_label = 0 if msg['sender_annotation'] == True else 1
    receiver_label = 0 if msg['receiver_annotation'] == True else 1
    
    if sender_label == 0:  # True sender annotation
        if receiver_label == 0:
            total_tt += 1
        else:
            total_tn += 1
    else:  # False sender annotation
        if receiver_label == 0:
            total_nt += 1
        else:
            total_nn += 1
            
    return sender_label, receiver_label


with jsonlines.open(test_set, 'r') as reader:
    train = list(reader)
    
    for msg in aggregate(train):
        labels = process_message(msg)
        if labels[0] is not None:
            sender_labels.append(labels[0])
            receiver_labels.append(labels[1])



print('Human baseline, macro:', f1_score(sender_labels, receiver_labels, pos_label=1, average= 'macro'))
print('Human baseline, lie F1:', f1_score(sender_labels, receiver_labels, pos_label=1, average= 'binary'))
print('Overall Accuracy is, ', accuracy_score(sender_labels, receiver_labels))

# Human baseline, macro: 0.5814484420580899 actual 
# Human baseline, lie F1: 0.22580645161290322 actual
# Overall Accuracy is,  0.8836363636363637
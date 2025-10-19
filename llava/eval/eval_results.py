import pandas as pd
import pickle
import json
from PIL import Image
from tqdm.notebook import tqdm
import os
import base64
import requests
import re
import collections
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import numpy as np
from io import BytesIO
import argparse

def classification_metrics(y_true, y_pred):

    unique_labels = set.union(set(y_true), set(y_pred))
    exclude_class = len(set(y_true))
    labels_to_consider = [label for label in unique_labels if label != exclude_class]
    print('gt labels: ', labels_to_consider)
    print('exclude labels: ', exclude_class)

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(y_true, y_pred, average=None, labels=labels_to_consider)
    recall = recall_score(y_true, y_pred, average=None, labels=labels_to_consider)
    f1 = f1_score(y_true, y_pred, average=None, labels=labels_to_consider)

    macro_precision = precision_score(y_true, y_pred, average='macro', labels=labels_to_consider)
    macro_recall = recall_score(y_true, y_pred, average='macro', labels=labels_to_consider)
    macro_f1 = f1_score(y_true, y_pred, average='macro', labels=labels_to_consider)

    micro_precision = precision_score(y_true, y_pred, average='micro', labels=labels_to_consider)
    micro_recall = recall_score(y_true, y_pred, average='micro', labels=labels_to_consider)
    micro_f1 = f1_score(y_true, y_pred, average='micro', labels=labels_to_consider)
    
    if len(set(y_true)) == 2:
        AUC_cls = roc_auc_score(y_true, y_pred)
    else:
        AUC_cls = -1
    cm = confusion_matrix(y_true, y_pred)

    return {
        'accuracy': accuracy,
        'roc_auc': AUC_cls,
        'macro_f1': macro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'micro_f1': micro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'f1_per_class': f1,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'confusion_matrix': cm
    }

def print_metrics(metrics_dict, print_all=False):
    print(f"{metrics_dict['accuracy']*100 :.2f}", end='| ')
    print(f"real acc {metrics_dict['recall_per_class'][0]*100 :.2f}", end='| ')
    print(f"fake acc {metrics_dict['recall_per_class'][1]*100 :.2f}", end='| ')
    print(f"real f1 {metrics_dict['f1_per_class'][0]*100 :.2f}", end='| ')
    print(f"fake f1 {metrics_dict['f1_per_class'][1]*100 :.2f}")
    if print_all:
        for key, value in metrics_dict.items():
            if key in [
                    'confusion_matrix', 'f1_per_class', 'precision_per_class',
                    'recall_per_class'
            ]:
                print(f"{key}: {value}")
            else:
                print(f"{key}: {value*100 :.2f}")

def eval_task(judge_file):
    data_J = {}
    question_J = {}
    gts = {}
    unknown_cases_1 =[]
    with open(judge_file, "r") as r_file:
        for line_i, line in enumerate(r_file):
            if line:
                temp = json.loads(line)
                if "question_id" in temp.keys():
                    line_id = "question_id"
                elif "idx" in temp.keys():
                    line_id = "idx"
                elif "id" in temp.keys():
                    line_id = "id"
                data_J[temp[line_id]] = temp["choices"][0]["turns"]
                question_J[temp[line_id]] = temp["turns"][0]
                gts[temp[line_id]] = temp["reference"]

                
    o_dict = collections.defaultdict(list)
    for k, v in data_J.items():
        o_dict['idx'].append(k)
        o_dict['ground_truth'].append(gts[k])
        answer_J_1 = data_J[k][0]
        predict_label_1 = answer_J_1.split('\n')[-1].lower().strip()
        
        if re.search(r'fake', predict_label_1, re.IGNORECASE):
            o_dict['pred'].append('fake')
        elif re.search(r'real', predict_label_1, re.IGNORECASE):
            o_dict['pred'].append('real')
        elif re.search(r'yes', predict_label_1, re.IGNORECASE):
            o_dict['pred'].append('fake')
        elif re.search(r'no', predict_label_1, re.IGNORECASE):
            o_dict['pred'].append('real')
        elif re.search(r'fake', answer_J_1, re.IGNORECASE):
            o_dict['pred'].append('fake')
        elif re.search(r'yes', answer_J_1, re.IGNORECASE):
            o_dict['pred'].append('fake')
        elif re.search(r'real', answer_J_1, re.IGNORECASE):
            o_dict['pred'].append('real')
        elif re.search(r'no', answer_J_1, re.IGNORECASE):
            o_dict['pred'].append('real')
        else:
            unknown_cases_1.append(k)  
            o_dict['pred'].append('failed')
        
        o_dict['question'].append(question_J[k])
        o_dict['answer'].append(answer_J_1)
        
        
    df_o = pd.DataFrame(o_dict)
    
    y_true = df_o["ground_truth"].map({"fake": 1, "real": 0, "failed": 2})
    y_pred = df_o["pred"].map({"fake": 1, "real": 0, "failed": 2})
    print(y_true.value_counts())
    print(len(y_true), len(y_pred))
    print_metrics(classification_metrics(y_true, y_pred), True)
    
    return df_o    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge_file", type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    eval_task(args.judge_file)

import os
import json
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import Dataset
from config import DefaultConfig


class TextDataset(Dataset):
    def __init__(self, data_name, model_name="gpt"):
        self.data_name = data_name
        self.model_name = model_name
        (self.X, self.y,
         self.normal_label_list, self.anomaly_label_list, 
         self.origianl_task, 
         self.normal_desc_dict, self.anomaly_desc_dict) = self._process_dataset()

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx]
    
    def get_labels(self):
        return self.y
    
    def get_normal_label_list(self):
        return self.normal_label_list
    
    def get_anomaly_label_list(self):
        return self.anomaly_label_list
    
    def get_origianl_task(self):
        return self.origianl_task
    
    def get_normal_desc_dict(self):
        return self.normal_desc_dict
    
    def get_anomaly_desc_dict(self):
        return self.anomaly_desc_dict
    
    def get_X(self):
        return self.X

    def _process_dataset(self):
        cur_dir = os.path.dirname(__file__)
        data_dir = os.path.join(cur_dir, 'data')
        file_path = os.path.join(data_dir, self.data_name, 
                                 f"{self.data_name}_test_data.jsonl")
        X, y = [], []
        normal_label_list, anomaly_label_list = [], []
        normal_desc_dict, anomaly_desc_dict = {}, {}
        origianl_task = None
        print(f"Start reading dataset {self.data_name}...")
        with open(file_path, 'r') as file:
            for line in file:
                try:
                    data = json.loads(line)
                    text, label = data['text'], data['label']
                    if label != 0 and label != 1:
                        raise ValueError("Invalid label value.")
                    X.append(text)
                    y.append(label)
                except json.JSONDecodeError:
                    print(f"!!! Error reading line: {line} in {file_path}")
                    continue
        
        data_sum_path = os.path.join(data_dir, "data_summary.jsonl")
        with open(data_sum_path, 'r') as file:
            for line in file:
                try:
                    data = json.loads(line)
                    if data['name'] == self.data_name:
                        normal_label_list = data['normal_label_list']
                        anomaly_label_list = data['anomaly_label_list']
                        origianl_task = data['origianl_task']
                        break
                except json.JSONDecodeError:
                    print(f"!!! Error reading line: {line} in {data_sum_path}")
                    continue

        if not DefaultConfig._use_desc:
            print("Complete reading data.")
            return X, y, normal_label_list, anomaly_label_list, origianl_task, \
                normal_desc_dict, anomaly_desc_dict
        
        data_desc_path = os.path.join(data_dir, self.data_name,
                                      f"{self.data_name}_{self.model_name}_desc.json")
        with open(data_desc_path, 'r') as file:
            try:
                data = json.load(file)
                # print(data)
            except json.JSONDecodeError:
                raise ValueError(f"!!! Error reading {data_desc_path}")
            
        # get normal_desc_dict and anomaly_desc_dict
        for key, value in data.items():
            if key in normal_label_list:
                normal_desc_dict[key] = value
            elif key in anomaly_label_list:
                anomaly_desc_dict[key] = value
            else:
                raise ValueError(f"!!! Error: {key} not in normal or anomaly label list")
        
        print("Complete reading data.")
        return X, y, normal_label_list, anomaly_label_list, origianl_task, \
            normal_desc_dict, anomaly_desc_dict


class SynthDataset(Dataset):
    def __init__(self, data_name, mode=0, model_name="gpt"):
        self.data_name = data_name
        self.part_X = self._process_part_dataset()
        self.mode = mode
        self.model_name = model_name
        # mode 0: part dataset
        # mode 1: synth + part dataset
        # mode 2: synth dataset
        # check the mode
        if self.mode == 0:
            self.X = self.part_X
        elif self.mode == 1:
            self.synth_X = self._process_synth_dataset()
            self.X = self.part_X + self.synth_X
        elif self.mode == 2:
            self.X = self._process_synth_dataset()
        else:
            raise ValueError("Invalid mode value.")

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx]

    def _process_part_dataset(self):
        cur_dir = os.path.dirname(__file__)
        data_dir = os.path.join(cur_dir, 'data')
        file_path = os.path.join(data_dir, self.data_name, 
                                 f"{self.data_name}_train_part_data.jsonl")
        X = []
        with open(file_path, 'r') as file:
            for line in file:
                try:
                    data = json.loads(line)
                    text = data['text']
                    X.append(text)
                except json.JSONDecodeError:
                    print(f"!!! Error reading line: {line} in {file_path}")
                    continue
        return X
    
    def _process_synth_dataset(self):
        cur_dir = os.path.dirname(__file__)
        data_dir = os.path.join(cur_dir, 'data')
        use_desc = ""
        if DefaultConfig._use_desc:
            use_desc = "_use_desc"
        file_path = os.path.join(data_dir, self.data_name, 
                                 f"{self.data_name}_{self.model_name}_synth_data{use_desc}.jsonl")
        X = []
        with open(file_path, 'r') as file:
            for line in file:
                try:
                    data = json.loads(line)
                    text = data['text']
                    X.append(text)
                except json.JSONDecodeError:
                    print(f"!!! Error reading line: {line} in {file_path}")
                    continue
        return X


def evaluate(y_true, y_score, threshold=0.5):
    sample_count = len(y_true)
    if not isinstance(y_score, np.ndarray):
        y_score = np.array(y_score)
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)

    # delete error samples
    print(f"Error count: {np.sum(y_score == DefaultConfig.error_symbol)}")
    error_indecies = np.where(y_score == DefaultConfig.error_symbol)
    y_true = np.delete(y_true, error_indecies)
    y_score = np.delete(y_score, error_indecies)
    
    tp = np.sum((y_true == 1) & (y_score >= threshold))
    fp = np.sum((y_true == 0) & (y_score >= threshold))
    tn = np.sum((y_true == 0) & (y_score < threshold))
    fn = np.sum((y_true == 1) & (y_score < threshold))

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        print("ZeroDivisionError: precision")
        precision = 0
    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        print("ZeroDivisionError: recall")
        recall = 0
    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        print("ZeroDivisionError: f1")
        f1 = 0
    
    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)

    print("Evaluation results:")
    print(f"AUROC: {auroc}")
    print(f"AUPRC: {auprc}")
    print("=====================================")
    print(f"The following results are calculated based on threshold={threshold}.")
    print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")


def save_json(generated_json, data_name, suffix):
    cur_dir = os.path.dirname(__file__)
    data_dir = os.path.join(cur_dir, 'data')
    file_path = os.path.join(data_dir, data_name, 
                             f"{data_name}_{suffix}.json")
    with open(file_path, 'w') as file:
        json.dump(generated_json, file, indent=4)
    print(f"Saved the generated JSON to {file_path}")


def read_data_summary(data_name):
    cur_dir = os.path.dirname(__file__)
    data_dir = os.path.join(cur_dir, 'data')
    file_path = os.path.join(data_dir, "data_summary.jsonl")
    size = 0
    normal_label_list, anomaly_label_list = [], []
    origianl_task = None
    with open(file_path, 'r') as file:
            for line in file:
                try:
                    data = json.loads(line)
                    if data['name'] == data_name:
                        size = data['size']
                        normal_label_list = data['normal_label_list']
                        anomaly_label_list = data['anomaly_label_list']
                        origianl_task = data['origianl_task']
                        break
                except json.JSONDecodeError:
                    print(f"!!! Error reading line: {line} in {file_path}")
                    continue
    return normal_label_list, anomaly_label_list, origianl_task, size


def read_json(data_name, file_name):
    cur_dir = os.path.dirname(__file__)
    data_dir = os.path.join(cur_dir, 'data')
    file_path = os.path.join(data_dir, data_name, file_name)
    with open(file_path, 'r') as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            raise ValueError(f"!!! Error reading {file_path}")
    return data


def read_normal_desc(data_name, model_name, normal_label_list):
    file_name = f"{data_name}_{model_name}_desc.json"
    data = read_json(data_name, file_name)
    normal_desc_dict = {}
    
    # get normal_desc_dict
    for key, value in data.items():
        if key in normal_label_list:
            normal_desc_dict[key] = value
        else:
            # abnormal label
            continue
    return normal_desc_dict

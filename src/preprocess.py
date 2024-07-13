import ast
import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoTokenizer

import config




def myfunction(df):

    x = ast.literal_eval(df)
    x = [n.strip() for n in x]

    return x


def Convert(string):
    li = list(string.split(" "))
    li = "#".join(li).upper()
    # print(li)
    return li




def process_data(dir, file, RelationLevl):

    df = pd.read_csv(os.path.join(dir, file + ".csv"))
    if RelationLevl == 1:
        df["Com2"] = df["emd_con1"].apply(lambda x: myfunction(x))
    elif RelationLevl == 2:
        df["Com2"] = df["emd_con2"].apply(lambda x: myfunction(x))
    elif RelationLevl == 3:
        df["Com2"] = df["emd_con3"].apply(lambda x: myfunction(x))
    elif RelationLevl == 4:
        df["Com2"] = df["emd_con4"].apply(lambda x: myfunction(x))
    # code block 1
    df["opinion_type"] = df["opinion_type"].apply(lambda x: Convert(x))
    df["Aspect"] = df["aspect_p"].apply(lambda x: Convert(x))
    df["guid"] = df["guid"]
    df["sentence"] = df["sentence"].apply(lambda x: myfunction(x))
    df["Tag"] = df["Target_Tag"].apply(lambda x: myfunction(x))

    data = df[["guid", "sentence", "Aspect", "Tag", "Com2"]].copy()
    return data


def Pad_labels(label, maxlength, pad=-100):
    if len(label) >= maxlength:
        return label[:maxlength]
    return label + ([pad] * (maxlength - len(label)))


def tokenize_and_align_com(text, tags):
    label_all_tokens = False
    tokenized_inputs = config.tokenizerFast(
        text, truncation=True, is_split_into_words=True
    )

    word_ids = tokenized_inputs.word_ids()
    labels = []

    previous_word_idx = None
    label_ids = []
    for word_idx in word_ids:  # Set the special tokens to -100.
        if word_idx is None:
            label_ids.append(0)
        elif (
            word_idx != previous_word_idx
        ):  # Only label the first token of a given word.
            label_ids.append(tags[word_idx])
        else:
            label_ids.append(tags[word_idx] if label_all_tokens else 0)
        previous_word_idx = word_idx
    labels = label_ids

    # tokenized_inputs["labels"] = labels
    return labels


def tokenize_and_align_tags(text, tags):
    label_all_tokens = False
    tokenized_inputs = config.tokenizerFast(
        text, truncation=True, is_split_into_words=True
    )

    word_ids = tokenized_inputs.word_ids()
    labels = []

    previous_word_idx = None
    label_ids = []
    for word_idx in word_ids:  # Set the special tokens to -100.
        if word_idx is None:
            label_ids.append("O")
        elif (
            word_idx != previous_word_idx
        ):  # Only label the first token of a given word.
            label_ids.append(tags[word_idx])
        else:
            label_ids.append(tags[word_idx] if label_all_tokens else 0)
        previous_word_idx = word_idx
    labels = label_ids

    # tokenized_inputs["labels"] = labels
    return labels


def tokenize_and_align_tags_mask(text, tags, special_tokens):
    label_all_tokens = False
    tokenized_inputs = config.tokenizerFast(
        text, truncation=True, is_split_into_words=True
    )

    word_ids = tokenized_inputs.word_ids()
    labels = []

    previous_word_idx = None
    label_ids = []
    for word_idx in word_ids:  # Set the special tokens to -100.
        if word_idx is None:
            label_ids.append(special_tokens)
        elif (
            word_idx != previous_word_idx
        ):  # Only label the first token of a given word.
            label_ids.append(tags[word_idx])
        else:
            label_ids.append(tags[word_idx] if label_all_tokens else special_tokens)
        previous_word_idx = word_idx
    labels = label_ids

    # tokenized_inputs["labels"] = labels
    return labels


def tokenize_and_align_com_mask(text, tags, special_tokens):
    label_all_tokens = False
    tokenized_inputs = config.tokenizerFast(
        text, truncation=True, is_split_into_words=True
    )

    word_ids = tokenized_inputs.word_ids()
    labels = []

    previous_word_idx = None
    label_ids = []
    for word_idx in word_ids:  # Set the special tokens to -100.
        if word_idx is None:
            label_ids.append(special_tokens)
        elif (
            word_idx != previous_word_idx
        ):  # Only label the first token of a given word.
            label_ids.append(tags[word_idx])
        else:
            label_ids.append(tags[word_idx] if label_all_tokens else special_tokens)
        previous_word_idx = word_idx
    labels = label_ids

    # tokenized_inputs["labels"] = labels
    return labels


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        self.guid = dataframe["guid"]
        self.sentence = dataframe["sentence"]
        self.aspect = dataframe["Aspect"]
        self.tag = dataframe["Tag"]
        self.com = dataframe["Com2"]

    def __len__(self):
        return len(self.guid)

    def __getitem__(self, idx):
        guid = self.guid.iloc[idx]
        sentence = [i.replace("'", "").strip("][") for i in self.sentence.iloc[idx]]
        # aspect = [i.replace("'", "").strip("][") for i in self.aspect.iloc[idx]]
        aspect = [self.aspect.iloc[idx]]
        # print(aspect)
        tag = [
            i.replace("'", "").strip("][") for i in str(self.tag.iloc[idx]).split(", ")
        ]
        com = [
            i.replace("'", "").strip("][") for i in str(self.com.iloc[idx]).split(", ")
        ]

        tag_labels = []
        labels_align = tokenize_and_align_tags(sentence, tag)
        # print(labels_align)
        # print(tag)
        tag_labels = [config.TAG2ID[i] if i != 0 else 0 for i in labels_align]
        # tagorginal=[config.TAG2ID[i]    for i in labels_align if i != config.special_tokens]
        orginal_inputs = config.tokenizerFast(
            sentence, truncation=True, is_split_into_words=True
        )
        orginal_input = len(orginal_inputs["input_ids"])
        # print(orginal_inputs['input_ids'])
        # print(tag_labels)
        input_len = len(tag_labels)
        labels_com = []
        align_coms = tokenize_and_align_com(sentence, com)
        labels_com = [config.COM2ID[ii] if ii != 0 else 0 for ii in align_coms]
        labels_com2 = [1 if ii != 0 else 0 for ii in labels_com]
        # print(labels_com2)
        labels_com_Pad = [config.COM2ID[ii] if ii != 0 else 0 for ii in align_coms]
        labels_mask = tokenize_and_align_tags_mask(sentence, tag, config.special_tokens)

        # tag_mask_len = [1  for i in tag_labels if tag_labels[i] !=0 else 0  ]
        tag_mask_len = [1 for i in labels_mask]
        # tag_mask_len2 = [1 for i in labels_mask_2]
        tag_mask = Pad_labels(tag_mask_len, config.MAX_LEN, 0)
        tag_mask2 = Pad_labels(labels_com2, config.MAX_LEN, 0)
        # print('tag_mask',tag_mask)

        # print('tag_mask2',tag_mask2)
        # print(tag_mask)
        # tag_mask2 = [0  for i in tag_mask  if tag_mask[i] = 0]
        # print(tag_mask2)
        # print(tag_mask)
        # print("sentence", sentence)
        tokenized_input_full = config.tokenizerFast(
            sentence,
            aspect,
            truncation=True,
            max_length=config.MAX_LEN,
            padding="max_length",
            is_split_into_words=True,
        )
        # print(aspect)
        Aspect_label = [config.ASPECT2ID[ii] for ii in aspect]
        # print(Aspect_label)
        Pad_Tags = Pad_labels(tag_labels, config.MAX_LEN, 0)
        # print(Pad_Tags)
        # print( tokenized_input_full["input_ids"])
        # print(config.tokenizerFast.decode(tokenized_input_full["input_ids"]))
        # assert 1==5
        Pad_Com = Pad_labels(labels_com, config.MAX_LEN)
        Pad_Com_Zero = Pad_labels(labels_com_Pad, config.MAX_LEN, 0)
        # print(tag_mask)
        # new_pad = Pad_Com[Pad_Com == -100] = 0
        # print(new_pad)
        tokenized_input_full["tag_label_id"] = Pad_Tags

        # tokenized_input_full["com_label_id"] = new_pad
        # print(Pad_Tags)
        return {
            "input_ids": torch.tensor(
                tokenized_input_full["input_ids"], dtype=torch.long
            ),
            "token_type_ids": torch.tensor(
                tokenized_input_full["token_type_ids"], dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                tokenized_input_full["attention_mask"], dtype=torch.bool
            ),
            "tag_label_id": torch.tensor(Pad_Tags, dtype=torch.long),
            "com_label_id": torch.tensor(Pad_Com_Zero, dtype=torch.long),
            "tag_mask": torch.tensor(tag_mask, dtype=torch.long),
            "input_len": torch.tensor(input_len, dtype=torch.long),
            "aspect_label": torch.tensor(Aspect_label, dtype=torch.long),
            "sentence_org": torch.tensor(orginal_input, dtype=torch.long),
        }


class MetricsTracking:
    """
  In order make the train loop lighter I define this class to track all the metrics that we are going to measure for our model.
  """

    def __init__(self):

        self.total_acc = 0
        self.total_f1 = 0
        self.total_precision = 0
        self.total_recall = 0

    def update(self, predictions, labels, ignore_token=0):
        """
    Call this function every time you need to update your metrics.
    Where in the train there was a -100, were additional token that we dont want to label, so remove them.
    If we flatten the batch its easier to access the indexed = -100
    """
        predictions = predictions.flatten()
        labels = labels.flatten()

        # predictions=flatten_list(predictions)

        predictions = predictions[labels != ignore_token]
        labels = labels[labels != ignore_token]

        # predictions = predictions.to("cpu")
        # labels = labels.to("cpu")

        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="macro", zero_division=0)
        precision = precision_score(
            labels, predictions, average="macro", zero_division=0
        )
        recall = recall_score(labels, predictions, average="macro", zero_division=0)

        self.total_acc += acc
        self.total_f1 += f1
        self.total_precision += precision
        self.total_recall += recall

    def updateCRF(self, predictions, labels, ignore_token=0):
        """
    Call this function every time you need to update your metrics.
    Where in the train there was a -100, were additional token that we dont want to label, so remove them.
    If we flatten the batch its easier to access the indexed = -100
    """
        # predictions = predictions.flatten()
        # labels = labels.flatten()

        # predictions=flatten_list(predictions)

        # predictions = predictions[labels != ignore_token]
        # labels = labels[labels != ignore_token]

        # predictions = predictions.to("cpu")
        # labels = labels.to("cpu")

        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="macro", zero_division=0)
        precision = precision_score(
            labels, predictions, average="macro", zero_division=0
        )
        recall = recall_score(labels, predictions, average="macro", zero_division=0)

        self.total_acc += acc
        self.total_f1 += f1
        self.total_precision += precision
        self.total_recall += recall

    def return_avg_metrics(self, data_loader_size):
        n = data_loader_size
        metrics = {
            "acc": round(self.total_acc / n, 3),
            "f1": round(self.total_f1 / n, 3),
            "precision": round(self.total_precision / n, 3),
            "recall": round(self.total_recall / n, 3),
        }
        return metrics


def flat_accuracy(preds, labels):
    ignore_token = -100
    preds = preds[labels != ignore_token]
    labels = labels[labels != ignore_token]
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def flatten_list(_2d_list):
    flat_list = []
    # myarray = np.empty(shape, dtype=np.int)
    # Iterate through the outer list
    for element in _2d_list:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                if item != 0:
                    flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list


def compute_acc_recall(batch_output, batch_tag):
    acc = 0
    recall = 0
    f1 = 0
    precision = 0
    # print(batch_output,"batch_output")
    # print(batch_tag,"batch_tag")
    for index in range(len(batch_output)):
        acc += accuracy_score(batch_output[index], batch_tag[index])
        recall += recall_score(
            batch_output[index], batch_tag[index], average="macro", zero_division=0
        )
        f1 += f1_score(
            batch_output[index], batch_tag[index], average="macro", zero_division=0
        )
        precision += precision_score(
            batch_output[index], batch_tag[index], average="macro", zero_division=0
        )
    return (
        round(acc / len(batch_output), 3),
        round(recall / len(batch_output), 3),
        round(f1 / len(batch_output), 3),
        round(precision / len(batch_output), 3),
    )


def save_metrics(
    path,
    train_loss_list,
    valid_loss_list,
    epochs,
    accuracy,
    f1_scors,
    precisions,
    recalls,
    As_accuracy,
    As_f1_scors,
    As_precisions,
    As_recalls,
    Dataset,
):
    state_dict = {
        "train_loss_list": train_loss_list,
        "valid_loss_list": valid_loss_list,
        "epochs": epochs,
        "accuracy": accuracy,
        "f1_scors": f1_scors,
        "precisions": precisions,
        "recalls": recalls,
        "As_accuracy": As_accuracy,
        "As_f1_scors": As_f1_scors,
        "As_precisions": As_precisions,
        "As_recalls": As_recalls,
        "Dataset": Dataset,
    }

    torch.save(state_dict, path)


def load_metrics(path):
    device = torch.device("cuda")
    state_dict = torch.load(path, map_location=device)
    return (
        state_dict["train_loss_list"],
        state_dict["valid_loss_list"],
        state_dict["epochs"],
        state_dict["accuracy"],
        state_dict["f1_scors"],
        state_dict["precisions"],
        state_dict["recalls"],
        state_dict["As_accuracy"],
        state_dict["As_f1_scors"],
        state_dict["As_precisions"],
        state_dict["As_recalls"],
        state_dict["Dataset"],
    )


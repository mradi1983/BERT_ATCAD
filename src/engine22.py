import re

import numpy as np
import torch
import torch.nn.functional as FF
from sklearn_crfsuite import metrics
from torch import nn
from tqdm import tqdm

import config
from preprocess import MetricsTracking, compute_acc_recall


def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    entity_label = r"BI*"  # for BIO
    final_loss = 0
    batch_index = 0
    nb_tr_examples = 0
    losses = []
    correct = 0
    tr_ner_loss = 0
    correct_predictions = 0
    Gold_Num = 0
    True_Num = 0
    Pre_Num = 0
    N = 0
    for batch in tqdm(data_loader, total=len(data_loader)):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        tag_label_id = batch["tag_label_id"].to(device)
        com_label_id = batch["com_label_id"].to(device)
        tag_mask = batch["tag_mask"].to(device)
        aspect_label = batch["aspect_label"].to(device)
        sentence_org = batch["sentence_org"].to(device)

        optimizer.zero_grad()

        ner_logits, aspect_out, loss = model(
            input_ids,
            attention_mask,
            token_type_ids,
            com_label_id,
            tag_label_id,
            tag_mask,
            aspect_label,
            sentence_org,
        )
        loss.backward()

        nb_tr_examples += input_ids.size(0)
        tr_ner_loss += loss.item()

      
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    return round((tr_ner_loss / nb_tr_examples) * 100, 2) 


def eval_fn(data_loader, model, device):
    model.eval()
    pred_tag = []
    true_tag = []
    Acc = 0
    dev_metrics = MetricsTracking()
    
 
    entity_label = r"BI*"  # for BIO
    Gold_Num = 0
    True_Num = 0
    Pre_Num = 0
    correct = 0
    N = 0
    nb_test_examples = 0
   
    ner_test_loss = 0
    batch_index = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            tag_label_id = batch["tag_label_id"].to(device)
            com_label_id = batch["com_label_id"].to(device)
            tag_mask = batch["tag_mask"].to(device)
            aspect_label = batch["aspect_label"].to(device)
            sentence_org = batch["sentence_org"].to(device)
            ner_logits, aspect_out, loss = model(
                input_ids,
                attention_mask,
                token_type_ids,
                com_label_id,
                tag_label_id,
                tag_mask,
                aspect_label,
                sentence_org,
            )

            batch_index += 1

            ner_logit = model.CRF_model.decode(
                ner_logits, mask=tag_mask.type(torch.uint8)
            )

            input_id = input_ids.to("cpu").numpy()

            sentehce_tokens_length = sentence_org.to("cpu").numpy()
            tag_label_id = tag_label_id.to("cpu").numpy()

            input_id = [[idx for idx in indices] for indices in input_id]
            ner_label_ids = [[idx for idx in indices] for indices in tag_label_id]
            ner_logits = [[idx for idx in indices] for indices in ner_logit]

            for output_i in range(len(input_id)):

                sentence_length = sentehce_tokens_length[output_i]
                sentence_ids = input_id[output_i]
                sentence = config.TOKENIZER.convert_ids_to_tokens(
                    sentence_ids[0:sentence_length]
                )

                sentence_clean = []
                label_true = []
                label_pre = []
                sentence_len = len(sentence)

                for i in range(sentence_len):
                    if not sentence[i].startswith("##"):
                        sentence_clean.append(sentence[i])
                        label_true.append(config.ID2TAG[ner_label_ids[output_i][i]])
                        label_pre.append(config.ID2TAG[ner_logits[output_i][i]])
                label_true_np = np.array(label_true)
                label_pre_np = np.array(label_pre)
                N += label_true_np.shape[0]
                correct += (label_true_np == label_pre_np).sum()
               
                pre_ner_tags = "".join(label_pre)  # [CLS] sentence [SEP] ........
                gold_ner_tags = "".join(label_true)
                gold_entity = []
                pre_entity = []
                gold_entity_list = re.finditer(entity_label, gold_ner_tags)
                pre_entity_list = re.finditer(entity_label, pre_ner_tags)
                for x in gold_entity_list:
                    gold_entity.append(str(x.start()) + "-" + str(len(x.group())))
                for x in pre_entity_list:
                    pre_entity.append(str(x.start()) + "-" + str(len(x.group())))
                Gold_Num += len(gold_entity)
                Pre_Num += len(pre_entity)

                for x in gold_entity:
                    if x in pre_entity:
                        True_Num += 1
            ner_test_loss += loss.item()

            nb_test_examples += input_ids.size(0)

    Acc = correct / N
    P = True_Num / float(Pre_Num) if Pre_Num != 0 else 0
    R = True_Num / float(Gold_Num)
    F = (2 * P * R) / float(P + R) if P != 0 else 0
    # print("ASD task:")
    # print("\tP: ", P, "   R: ", R, "  F1: ", F)
    # print("----------------------------------------------------\n\n")

    # dev_metrics_ner.update(ner_out, label_ids)
    torch.cuda.empty_cache()

    assert len(pred_tag) == len(true_tag)
    metrics = {
        "acc": round(Acc * 100, 2),
        "f1": round(F * 100, 2),
        "precision": round(P * 100, 2),
        "recall": round(R * 100, 2),
    }
    loss = ner_test_loss / nb_test_examples

    dev_results = dev_metrics.return_avg_metrics(len(data_loader))
   
    return round(loss * 100, 2), metrics, dev_results


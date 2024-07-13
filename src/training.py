import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup

import config
import engine22
# from new_model_BERT_CRF import EntityModel
from new_model import EntityModel
from preprocess import (CustomDataset, MetricsTracking, flat_accuracy,
                        process_data, save_metrics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="input/Res16",
        type=str,
        required=False,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--RelationLevl",
        default=1,
        type=int,
        required=False,
        help="dependency relation level  1 , 2 ,3 , 4",
    )
    args = parser.parse_args()
    tarin_examples = process_data(args.data_dir, "train_new", args.RelationLevl)
    dev_examples = process_data(args.data_dir, "test", args.RelationLevl)

    train_dataset = CustomDataset(tarin_examples)
    dev_dataset = CustomDataset(dev_examples)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=config.train_batch_size
    )
    # for i in train_dataloader:
    #     print(i)

    dev_sampler = SequentialSampler(dev_dataset)
    dev_dataloader = torch.utils.data.DataLoader(
        dev_dataset, sampler=dev_sampler, batch_size=config.VALID_BATCH_SIZE
    )
    seed = 0  # https://arxiv.org/pdf/2109.08203.pdf 3407
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    # device = torch.device("cpu")
    model = EntityModel(num_tag=config.NUM_TAG, num_pos=config.NUM_COM, device=device)

    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "gamma", "beta"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay_rate": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay_rate": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5)

    num_train_steps = int(len(train_dataset) / config.train_batch_size * config.EPOCHS)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )
    train_loss_list = []
    valid_loss_list = []
    epochs = []
    accuracy = []
    f1_scors = []
    precisions = []
    recalls = []
    As_accuracy = []
    As_f1_scors = []
    As_precisions = []
    As_recalls = []
    Dataset = []

    curr_best_f1 = 0.0
    epocno = 0

    for epoch in range(config.EPOCHS):
        epocno += 1
        train_loss = engine22.train_fn(
            train_dataloader, model, optimizer, device, scheduler
        )
        dev_loss, dev_results, aspect_results = engine22.eval_fn(
            dev_dataloader, model, device
        )
        train_loss_list.append(train_loss)
        valid_loss_list.append(dev_loss)
        epochs.append(epocno)
        accuracy.append(dev_results["acc"])
        f1_scors.append(dev_results["f1"])
        precisions.append(dev_results["precision"])
        recalls.append(dev_results["recall"])
        As_accuracy.append(aspect_results["acc"])
        As_f1_scors.append(aspect_results["f1"])
        As_precisions.append(aspect_results["precision"])
        As_recalls.append(aspect_results["recall"])
        Dataset.append(config.TRAINING_FILE.split("/", 1)[1])
        print(
            f"Epochs: {epocno} |Train_Loss: {train_loss } | Val_Loss: {dev_loss} | acc: {dev_results['acc'] } | f1: {dev_results['f1'] } | precision: {dev_results['precision'] } | recall: {dev_results['recall'] }   | dataset: {args.data_dir.split('/',1)[1]} | RelationLevl: {args.RelationLevl  }   "
        )
        save_metrics(
            config.OUTPUTDIR
            + args.data_dir.split("/", 1)[1]
            + "/metric"
            + "_"
            + format(args.RelationLevl, "00")
            + ".pkl",
            train_loss_list,
            valid_loss_list,
            epochs,
            accuracy,
            f1_scors,
            precisions,
            recalls,
            As_f1_scors,
            As_accuracy,
            As_precisions,
            As_recalls,
            Dataset,
        )
        if dev_results["f1"] > curr_best_f1:
            # if dev_loss < best_loss:

            torch.save(
                model.state_dict(),
                config.OUTPUTDIR
                + args.data_dir.split("/", 1)[1]
                + "/"
                + args.data_dir.split("/", 1)[1]
                + "_"
                + format(args.RelationLevl, "00")
                + ".bin",
            )

            curr_best_f1 = dev_results["f1"]


if __name__ == "__main__":
    main()

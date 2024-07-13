import argparse
import os

import numpy as np
import torch

import config
from new_model import EntityModel
from preprocess import CustomDataset, process_data


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
    # test_examples = process_data(args.data_dir, "test", args.RelationLevl)
    test_examples = process_data(args.data_dir, "train_new", args.RelationLevl)
    # tarin_examples = process_data(args.data_dir, "train_new", args.RelationLevl)
    num_pos = config.NUM_COM
    num_tag = config.NUM_TAG
    # tarin_examples = process_data("train")

    # dev_examples = process_data("dev")
    # df = tarin_examples[tarin_examples["guid"] == "train-1122"]
    df = test_examples
    # objective function loop and predict for each sentence
    test_examples = CustomDataset(df)

    device = torch.device("cuda")
    model = EntityModel(num_tag=num_tag, num_pos=num_pos, device=device)
    MODEL_PATH = config.OUTPUTDIR+ args.data_dir.split("/", 1)[1]+ "/" + args.data_dir.split("/", 1)[1]+ "_"+ format(args.RelationLevl, '00') + ".bin"
   
    model.load_state_dict(torch.load(MODEL_PATH))
    # model.load_state_dict(torch.load(   model.state_dict()))
    model.to(device)
    with open(
        os.path.join( config.OUTPUTDIR
                + args.data_dir.split("/", 1)[1]
                + "/"
                + args.data_dir.split("/", 1)[1]
                + "_"
                 + "test"
                + format(args.RelationLevl, '00')
                + ".txt"),
        "w",
        encoding="utf-8",
    ) as f_test:
        f_test.write("As_true\tAs_pre\tsentence\ttrue_ner\tpredict_ner\n")
        for idx in range(len(test_examples)):
            with torch.no_grad():
                sentence = [
                    i.replace("'", "").strip("][") for i in df["sentence"].iloc[idx]
                ]
                tag_orginal = [
                    i.replace("'", "").strip("][") for i in df["Tag"].iloc[idx]
                ]
                asp_orginal = df["Aspect"].iloc[idx].replace("'", "").strip("][")

                batch = test_examples[idx]
                input_ids = batch["input_ids"].to(device).unsqueeze(0)
                attention_mask = batch["attention_mask"].to(device).unsqueeze(0)
                token_type_ids = batch["token_type_ids"].to(device).unsqueeze(0)
                tag_label_id = batch["tag_label_id"].to(device).unsqueeze(0)
                com_label_id = batch["com_label_id"].to(device).unsqueeze(0)
                tag_mask = batch["tag_mask"].to(device).unsqueeze(0)
                aspect_label = batch["aspect_label"].to(device).unsqueeze(0)
                sentence_org = batch["sentence_org"].to(device).unsqueeze(0)
                tag, aspect_out, _ = model(
                    input_ids,
                    attention_mask,
                    token_type_ids,
                    com_label_id,
                    tag_label_id,
                    tag_mask,
                    aspect_label,
                    sentence_org,
                )

            id2tag = {id: tag for id, tag in enumerate(config.TAG2ID)}
            id2asp = {id: asp for id, asp in enumerate(config.ASPECT2ID)}
            # id2tag[0] = "ignore"
            # print(tag)
            ner_logits = [
                id2tag[label.item()]
                for label in tag.argmax(2).cpu().numpy().reshape(-1)
            ]
            # print(aligned_labels)
            ner_label_ids = [
                id2tag[label] for label in tag_label_id.cpu().numpy().reshape(-1)
            ]

            aligned_labels_asp = [
                id2asp[label.item()]
                for label in aspect_out.argmax(1).cpu().numpy().reshape(-1)
            ]
            Aspect_pred = aligned_labels_asp[0]

            # ner_label_ids = [[idx for idx in indices] for indices in tag_label_id]
            # ner_logits = [[idx for idx in indices] for indices in ner_logit]
            input_ids = input_ids.to("cpu").numpy()
            input_id = np.array(input_ids[0])

            sentehce_tokens_length = sentence_org.to("cpu").numpy()
            sentence = config.TOKENIZER.convert_ids_to_tokens(
                input_id[0 : sentehce_tokens_length[0]]
            )

            label_true = []
            label_pre = []
            sentence_len = len(sentence)
            sentence_clean = config.TOKENIZER.decode(
                input_id[0 : sentehce_tokens_length[0]]
            )
            # print(sentence)
            for i in range(sentence_len):
                if not sentence[i].startswith("##"):
                    label_true.append(ner_label_ids[i])
                    label_pre.append(ner_logits[i])

            # label_true_np = np.array(label_true)
            # label_pre_np = np.array(label_pre)
            # print(label_true)
            # print(label_pre)

            f_test.write(str(asp_orginal))
            f_test.write("\t")
            f_test.write(str(Aspect_pred))
            f_test.write("\t")
            f_test.write(sentence_clean)
            f_test.write("\t")
            f_test.write(" ".join(label_true))
            f_test.write("\t")
            f_test.write(" ".join(label_pre))
            f_test.write("\n")
          

if __name__ == "__main__":
    main()
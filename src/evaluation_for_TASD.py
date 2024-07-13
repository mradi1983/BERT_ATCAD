# coding=utf-8

"""evaluate P R F1 for target & aspect - sentiment joint detection"""

import argparse
import os
import re

import config


def evaluate_TASD(path, dataset, level):
    with open(os.path.join(path), "r", encoding="utf-8") as f_pre:
        Gold_Num = 0
        True_Num = 0
        Pre_Num = 0
      
        f_pre.readline()
     
        entity_label = r"BI*"  # for BIO
        pre_lines = f_pre.readlines()
        for line in pre_lines:

            pre_line = line.strip().split("\t")
           
            pre_ner_tags = "".join(pre_line[4].split())  # [CLS] sentence [SEP] ........
            gold_ner_tags = "".join(pre_line[3].split())

            gold_aspect_tags = "".join(pre_line[0].split())
            pre_aspect_tags = "".join(pre_line[1].split())
         
            if gold_aspect_tags == pre_aspect_tags:  # 
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
            else:  # 
                Gold_Num += 1
                True_Num += 1
      

    P = True_Num / float(Pre_Num) if Pre_Num != 0 else 0
    R = True_Num / float(Gold_Num)
    F = (2 * P * R) / float(P + R) if P != 0 else 0
    print("\n\n")
    print("TASD task:  ", "Data: ",dataset,"  RelationLevel: ",level)
    print("----------------------------------------------------")
    print("\tP: ", "%.2f" % P, "   R: ", "%.2f" % R, "  F1: ", F)
    print("----------------------------------------------------\n\n")


if __name__ == "__main__":
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
  

    pred_path2 = os.path.join(config.OUTPUTDIR
                + args.data_dir.split("/", 1)[1]
                + "/"
                + args.data_dir.split("/", 1)[1]
                + "_"
                + format(args.RelationLevl, '00')
                + ".txt")
 
    evaluate_TASD(pred_path2 , args.data_dir.split("/", 1)[1],format(args.RelationLevl, '00'))
 
 


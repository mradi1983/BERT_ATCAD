# coding=utf-8

"""Processors for Semeval Dataset."""

import csv
import os

import pandas as pd
import spacy

import decomposition

# Load the installed model "en_core_web_sm"
nlp = spacy.load("en_core_web_sm")



class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(
        self,
        guid,
        text_a,
        aspect_p,
        opinion_type,
        bio_labels_it,
        bio_labels_et,
        target,
        opinin_Term,
        text_idx,
        text_head_idx,
        text_index,
        polarity,
        pos_tag,
        sentiment,
    ):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified. senetence
            aspect_p: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks. aspect & polarity
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
            ner_labels_a: ner tag sequence for text_a. This should be
            specified for train and dev examples, but not for test examples.
        """

        self.guid = guid
        self.text_a = text_a
        self.aspect_p = aspect_p
        self.opinion_type = opinion_type
        self.bio_labels_it = bio_labels_it
        self.bio_labels_et = bio_labels_et
        self.target = target
        self.opinin_Term = opinin_Term
        self.text_head_idx = text_head_idx
        self.text_idx = text_idx
        self.text_index = text_index
        self.polarity = polarity
        self.pos_tag=pos_tag
        self.sentiment=sentiment


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def get_ner_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    # @classmethod
    # def _read_tsv(cls, input_file, quotechar=None):
    #     """Reads a tab separated value file."""
    #     with open(input_file, "r") as f:
    #         reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
    #         lines = []
    #         for line in reader:
    #             lines.append(line)
    #         return lines


class Semeval_Processor(DataProcessor):
    """Processor for the SemEval 2015 and 2016 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(
            os.path.join(data_dir, "train.tsv"), encoding="utf-8", sep=","
        )
        train_data = train_data.reset_index()
        # train_data.
        # train_data = train_data[train_data.iloc[1122]]
        # train_data2= train_data.loc[1122] 
        # train_data = self._create_examples(train_data, "train")
        return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = pd.read_csv(
            os.path.join(data_dir, "dev.tsv"), encoding="utf-8", sep=","
        )
        dev_data = dev_data.reset_index()
        return self._create_examples(dev_data, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        test_data = pd.read_csv(
            os.path.join(data_dir, "test.tsv"), encoding="utf-8", sep=","
        )
        test_data = test_data.reset_index()
        return self._create_examples(test_data, "test")

    def isNaN(self, string):

        return string != string

    def get_head_text(self, sentence):
        head_text = []
        text = []
        text_idx = []
        result = []
        Sentence_dict = {
            "text": [],
            "text_index": [],
            "text_head": [],
            "text_idx": [],
            "text_head_idx": [],
            "pos": [],
        }
        # my_dict = {"text":[],"text_head":[],"text_idx":[]};
        doc = nlp(sentence)
        [token.text for token in doc]
        for i, token in enumerate(doc):
            Sentence_dict["text"].append(str(token.text))
            Sentence_dict["text_index"].append(token.idx)
            Sentence_dict["text_head"].append(token.head.text)
            Sentence_dict["text_idx"].append(i)
            Sentence_dict["text_head_idx"].append(token.head.idx)
            Sentence_dict["pos"].append(token.pos_)

        # for head_text in Sentence_dict["text_head"]:
        #     for index, word in enumerate(Sentence_dict["text"]):
        #         if head_text == word:
        #             Sentence_dict["text_head_idx"].append(index)

        return Sentence_dict



    def _create_examples(self, train_data, set_type):
        """Creates examples."""
        examples = []
        df = train_data.reset_index()

        for index, row in df.iterrows():
            guid = "%s-%s" % (set_type, index)
            
            text_a = row["sentence"].split()  # sentence
            aspect_p = row["category"] + " " + row["polarity"]  # Aspect & polarity
            sentiment=row["polarity"] 
            opinion_type = row["target_type"]  # opinion type
            bio_labels_it = row["opinin_Term"]
            # implicit BIO-infrence
            n = self.get_head_text(row["sentence"])
            bio_labels_et = n
            # excplicit BIO-infrence
            target = row["target"]
            opinin_Term = row["opinin_Term"]
            polarity = row["polarity"]

            text_idx = n["text_idx"]
            text_head_idx = n["text_head_idx"]
            text_index = n["text_index"]
            pos_tag = n["pos"]

            # opinin_Term_idx=
            examples.append(
                InputExample(
                    guid=guid,
                    text_a=text_a,
                    aspect_p=aspect_p,
                    opinion_type=opinion_type,
                    bio_labels_it=bio_labels_it,
                    bio_labels_et=bio_labels_et,
                    target=target,
                    opinin_Term=opinin_Term,
                    text_idx=text_idx,
                    text_head_idx=text_head_idx,
                    text_index=text_index,
                    polarity=polarity,
                    pos_tag=pos_tag,
                    sentiment=sentiment
                )
            )
        return examples


# file_path = os.path.join("./Data", "Res16_ABSA_QUAD", "")
# default = ("Data/Res16_ABSA_QUAD/",)
file_path = os.path.join("./Data", "Res16_ABSA_QUAD", "")
default = ("Data/Res15_ABSA_QUAD/",)
processor = Semeval_Processor()
# label_list = processor.get_labels()
ner_label_list = processor.get_train_examples(file_path)
# )  # BIO or TO tags for ner entity get_dev_examples
 
# get_dev_examples
# ner_label_list = processor.get_dev_examples(file_path)
#  get_test_examples
# ner_label_list = processor.get_test_examples(file_path)
data_list = []
example = dict()
for (i, token) in enumerate(ner_label_list):
 
    example["guid"] = token.guid
    example["sentence"] = token.bio_labels_et["text"]
    # example["bio_labels_et"] = token.bio_labels_et

    example["opinion_type"] = token.opinion_type
    example["target"] = token.target
    example["opinin_Term"] = token.opinin_Term
    example["polarity"] = token.polarity
    example["aspect_p"] = token.aspect_p
    example["sentiment"] = token.sentiment
    example["pos_tag"] = token.pos_tag
    t = decomposition.InputExample(
        opinion_term=token.opinin_Term,
        text_a=token.bio_labels_et["text"],
        text_idx=token.bio_labels_et["text_idx"],
        text_head_idx=token.bio_labels_et["text_head_idx"],
        text_index=token.bio_labels_et["text_index"],
        opinion_type=token.opinion_type,
        target=token.target,
        pos_tag=token.bio_labels_et["pos"],
    )

    example["emd_con1"] = t.get_Con1_idx()
    example["emd_con2"] = t.get_Con2_idx()
    example["emd_con3"] = t.get_Con3_idx()
    example["emd_con4"] = t.get_Con4_idx()
    example["OP_Term_Tag"] = t.convert_BIO()
    example["Target_Tag"] = t.convert_BIO_Target()
    example["text_idx"] = token.bio_labels_et["text_index"],
    example["text_position"] = token.bio_labels_et["text_idx"],
    example["token"] = token.bio_labels_et["text"],
    example["token_head"] = token.bio_labels_et["text_head_idx"]
    data_list.append(example)
    example = {}
    # print(token.aspect_p)
    # print(token.aspect_p)
t = pd.DataFrame.from_dict(data_list)
t['pos_tag']

print( t["token"])
print( t["text_position"])
print( t["text_idx"])
print( t["token_head"])
print( t["emd_con1"])
print( t["emd_con2"])
print( t["emd_con3"])
print( t["emd_con4"])
print( t["Target_Tag"])
data_dir = os.path.join("./Data", "Res16_data")
t.to_csv(os.path.join(data_dir, r"example_distinct.csv"))



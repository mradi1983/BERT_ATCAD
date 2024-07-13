# coding=utf-8

"""Processors for Semeval Dataset."""

import csv
import os
import re

import numpy as np
import pandas as pd
import spacy

# Load the installed model "en_core_web_sm"
nlp = spacy.load("en_core_web_sm")


class InputExample:
    """A single training/test example for simple sequence classification."""

    def __init__(
        self,
        opinion_term,
        text_a,
        text_idx=None,
        text_head_idx=None,
        text_index=None,
        opinion_type=None,
        target=None,
        pos_tag=None,
    ):
        """Constructs a InputExample.

        Args:
            Term_idx: position of opinion term in the sentence array.
            text_a: string. The untokenized text of the first sequence. 
            text_idx: position of eacth word text in the sentence array.
            text_head_idx: position of word head in the sentence array.
        """
        self.opinion_term = opinion_term
        self.text_a = text_a
        self.text_idx = text_idx
        self.text_head_idx = text_head_idx
        self.text_index = text_index
        self.opinion_type = opinion_type
        self.target = target
        self.pos_tag = pos_tag

    # def __str__(self):
    #     return f"From str method of Test: a is {self.a}, b is {self.b}"
    def get_array_index(self):
        a = np.array(self.text_index)
        b = np.array(self.text_head_idx)
        c = np.array(self.text_a)
        d = np.array(self.pos_tag)
        A = np.vstack([a, b, c, d])
        return A

    def get_opinion_term_idx(self):
        conection1 = []
        arr = self.get_array_index()
        doc = nlp(self.opinion_term)
        [token.text for token in doc]
        for i, token in enumerate(doc):

            for ii, tokens in enumerate(arr[2]):
                if token.text == tokens:
                    int_array = np.int_(arr[0][ii])
                    word = arr[0][ii]
                    conection1.append(int_array)
                    continue
        return conection1

    def convert_BIO(self):
        op_type = self.opinion_type
        entity_label = r"T+"
        ner_tags = "".join(self.get_BIO_Infrence())
        entity_list = re.finditer(entity_label, ner_tags)

        BIO_tags = ["O"] * len(ner_tags)
        if op_type == "implicit":
            for x in entity_list:
                start = x.start()
                en_len = len(x.group())
                BIO_tags[start] = "B-IT"
                for m in range(start + 1, start + en_len):
                    BIO_tags[m] = "I-IT"
        else:
            for x in entity_list:
                start = x.start()
                en_len = len(x.group())
                BIO_tags[start] = "B-ET"
                for m in range(start + 1, start + en_len):
                    BIO_tags[m] = "I-ET"
        return BIO_tags

    def convert_BIO_Target(self):
        op_type = self.opinion_type
        entity_label = r"T+"
        entity_label_implicit = r"T+"
        ner_tags = "".join(self.get_BIO_Infrence_Target())
        ner_tags_implicit = "".join(self.get_BIO_Infrence())
        entity_list = re.finditer(entity_label, ner_tags)
        entity_list_implicit = re.finditer(entity_label_implicit, ner_tags_implicit)

        BIO_tags = ["O"] * len(ner_tags)
        if op_type == "explicit":
            for x in entity_list:
                start = x.start()
                en_len = len(x.group())
                BIO_tags[start] = "B-EO"
                for m in range(start + 1, start + en_len):
                    BIO_tags[m] = "I-EO"
        else:
            for xx in entity_list_implicit:
                start = xx.start()
                en_len = len(xx.group())
                BIO_tags[start] = "B-NULL"
                for m in range(start + 1, start + en_len):
                    BIO_tags[m] = "I-NULL"

        return BIO_tags

    def get_BIO_Infrence(self):
        arr = self.get_array_index()
        BIO_tags = ["O"] * len(arr[2])
        entity_label = self.opinion_term.strip().split()
        for token in entity_label:

            for ii, tokens in enumerate(arr[2]):

                if token == tokens:
                    BIO_tags[ii] = "T"

        return BIO_tags

    def get_BIO_Infrence_Target(self):
        op_type = self.opinion_type
        arr = self.get_array_index()
        BIO_tags = ["O"] * len(arr[2])
        if op_type == "explicit":
            entity_label = self.target.strip().split()
            for token in entity_label:

                for ii, tokens in enumerate(arr[2]):

                    if token == tokens:
                        BIO_tags[ii] = "T"

        return BIO_tags

    def get_head2_idx(self):
        conection1 = []
        # r = True
        # hidx = len(self.text_head_idx)
        # textidx = len(self.text_idx)
        # if hidx == textidx:
        #     r = True
        # else:
        #     r = False

        arr = self.get_array_index()
        int_text_index = np.int_(arr[0])
        int_headindex = np.int_(arr[1])
        for idx in self.get_head1_idx():
            for ii, tokens in enumerate(int_headindex):
                if idx == tokens:
                    # int_array = np.int_(int_text_index[ii])
                    conection1.append(int_text_index[ii])
                    continue
        new_list = [
            text_idx
            for text_idx in conection1
            if text_idx not in self.get_opinion_term_idx()
        ]

        return new_list

    def get_head3_idx(self):
        conection1 = []
        arr = self.get_array_index()
        int_text_index = np.int_(arr[0])
        int_headindex = np.int_(arr[1])
        for idx in self.get_head2_idx():
            for ii, word in enumerate(int_headindex):
                if idx == word:
                    # int_array = np.int_(int_text_index[ii])
                    conection1.append(int_text_index[ii])
                    continue
        new_list = [
            text_idx
            for text_idx in conection1
            if text_idx not in self.get_opinion_term_idx()
        ]
        return new_list

    def get_head4_idx(self):
        conection1 = []
        arr = self.get_array_index()
        int_text_index = np.int_(arr[0])
        int_headindex = np.int_(arr[1])
        for idx in self.get_head3_idx():
            for ii, word in enumerate(int_headindex):
                if idx == word:
                    # int_array = np.int_(int_text_index[ii])
                    conection1.append(int_text_index[ii])
                    continue
        new_list = [
            text_idx
            for text_idx in conection1
            if text_idx not in self.get_opinion_term_idx()
        ]
        return new_list

    def get_head1_idx(self):
        conection1 = []
        for idx in self.get_opinion_term_idx():
            # if idx not in self.text_a:
            for index, word in enumerate(self.text_index):
                if (
                    idx
                    == word
                    # and index not in self.get_opinion_term_idx()
                    # and index not in self.get_head3_idx()
                ):
                    # opinion_term_span.append(A[1][idx])
                    conection1.append(self.text_head_idx[index])

            return conection1

    def get_Con1_idx(self):
        arr = self.get_array_index()
        int_text_index = np.int_(arr[0])
        # int_headindex = np.int_(arr[1])
        con = np.concatenate((self.get_opinion_term_idx(), self.get_head1_idx()))
        consort = np.sort(con)
        b = consort.tolist()
        ner_tag_list = ["O"] * len(int_text_index)
        for i, word in enumerate(int_text_index):
              if word in b and arr[3][i] not in ['DET','X','SPACE','SYM','PUNCT']:
                ner_tag_list[i] = arr[3][i]
                continue

        return ner_tag_list

    def get_Con2_idx(self):
        arr = self.get_array_index()
        int_text_index = np.int_(arr[0])
        con = np.concatenate(
            (self.get_opinion_term_idx(), self.get_head1_idx(), self.get_head2_idx(),)
        )
        consort = np.sort(con)
        b = consort.tolist()
        ner_tag_list = ["O"] * len(self.text_idx)
        for i, word in enumerate(int_text_index):
              if word in b and arr[3][i] not in ['DET','X','SPACE','SYM','PUNCT']:
                ner_tag_list[i] = arr[3][i]
                continue

        return ner_tag_list

    def get_Con3_idx(self):
        arr = self.get_array_index()
        int_text_index = np.int_(arr[0])
        con = np.concatenate(
            (
                self.get_opinion_term_idx(),
                self.get_head1_idx(),
                self.get_head2_idx(),
                self.get_head3_idx(),
            ),
        )
        consort = np.sort(con)
        b = consort.tolist()
        ner_tag_list = ["O"] * len(self.text_idx)
        for i, word in enumerate(int_text_index):
              if word in b and arr[3][i] not in ['DET','X','SPACE','SYM','PUNCT']:
                ner_tag_list[i] = arr[3][i]
                continue

        return ner_tag_list

    def get_Con4_idx(self):
        arr = self.get_array_index()
        int_text_index = np.int_(arr[0])
        con = np.concatenate(
            (
                self.get_opinion_term_idx(),
                self.get_head1_idx(),
                self.get_head2_idx(),
                self.get_head3_idx(),
                self.get_head4_idx(),
            )
        )
        consort = np.sort(con)
        b = consort.tolist()
        ner_tag_list = ["O"] * len(self.text_idx)
        for i, word in enumerate(int_text_index):
            if word in b and arr[3][i] not in ['DET','X','SPACE','SYM','PUNCT']:
                ner_tag_list[i] = arr[3][i]
                continue
        return ner_tag_list


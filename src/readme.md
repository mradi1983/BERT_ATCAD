# Target-Aspect-Sentiment Joint Detection: Uncovering Explicit and Implicit Targets through Aspect-Target-Context-Aware Detection (ATCAD)

This repo contains the annotated data and code for our paper [Target-Aspect-Sentiment Joint Detection: Uncovering Explicit and Implicit Targets through Aspect-Target-Context-Aware Detection]


## Short Summary 
- We aim to tackle the Target aspect sentiment  (TASD) task: given a sentence, we predict opinion the tree elements `(target , aspect category, sentiment polarity)`

## Data
- This study adopted the opinion term annotations from [Aspect sentiment quad prediction as paraphrase generation]
- Prepare the dataset with Opinion Term Contextual Syntactic using run_Contextual_Syntactic.py. Note that the data is already prepared.

- We use , namely `Res15` and `Res16` under the `input` dir.
- Each data instance contains several columns, each serving a specific purpose:
    -guid: Unique identifier for each record.
    -sentence: The original sentence being analyzed.
    -opinion_type: Type of opinion (explicit or implicit).
    -target: The targeted in the sentence.
    -opinion_Term: The specific term expressing the opinion.
    -polarity: Sentiment polarity associated with the opinion term.
    -aspect_p: Category and sentiment of the aspect.
    -sentiment: Overall sentiment of the target - aspect.
    -pos_tag: Part-of-speech tags for each word in the sentence.
    -emd_con1, emd_con2, emd_con3, emd_con4: Contextual syntactic embeddings, for each Dependency Relation Level.
    -OP_Term_Tag: Tags for the opinion terms in the sentence.
    -Target_Tag: Tags for the target aspects in the sentence.


## Requirements

We highly recommend you to install the specified version of the following packages  in requirements.txt 


## Quick Start

- Set up the environment as described in the above section
- The config.py file identifies the model output directory and output settings.
- Run training.py using command run pyton file in terminal use Dir input/Res16 for Res16 dataset or input/Res15 for Res16 dataset. You can change the depndeny relation level by changinf the parameter --RelationLevl from 1 or 2 ,3 or 4 
- Run predict.py to obtain the result 




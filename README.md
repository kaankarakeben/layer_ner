# Fine Tuning BERT For Named Entity Recognition On United Nations Documents

[![Open in Layer](https://development.layer.co/assets/badge.svg)](https://app.layer.ai/kaankarakeben/united_nations_ner-finetuning)
[![Layer NER](https://badgen.net/badge/icon/github?icon=github&label)](https://github.com/kaankarakeben/layer_ner)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zkmCzzdOt4suAEPdQqN6uIWIHXTP-hNn?)

Humans understand the world by putting labels on things and examining how these labels relate to each other. A reflection of this in the natural language processing and information retrieval world is a technique called Named Entity Recognition (NER). The objective is to detect the entity type of segments of text in a document. These entities could be organizations, locations, persons, or others. 

In this blog post, I will go through an example of learning a named entity recognition model on a specific domain. Instead of creating a NER model from scratch, I will use transfer-learning by taking a pre-trained language model, BERT, trained on a large number of general examples and fine-tune that neural network on a very specific type of domain. 

Alongside the tutorial on learning a NER model, I will run this project on [Layer](https://layer.ai/) in order to make use of their metadata store for storing and tracking the datasets and model artifacts as well as their free GPU compute instances. 

Firstly, let's define the problem. We are working with a set of documents from the United Nations (UN). Diplomatic jargon is the norm at the UN and these documents contain many specific entities that we don't encounter in everyday language such as the Office for the Coordination of Humanitarian Affairs of the Secretariat or the Office of the United
Nations High Commissioner for Refugees. We would like to automatically detect these entities with their corresponding types. With the entities flagged, we can power many interesting use cases such as information retrieval, question/answering, document similarity, etc. 

## Installing and Importing Libraries

``` Bash
!pip install layer --upgrade -qqq
!pip install -U ipython

!pip install transformers
!pip install datasets
!pip install seqeval

!git clone https://github.com/leslie-huang/UN-named-entity-recognition
```

``` Python
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torch

import layer
from layer.decorators import dataset, model, pip_requirements, fabric, resources

layer.login()
layer.init("united-nations-ner-finetuning")

TRAIN_EXAMPLES_RATIO = 0.8
MAX_LEN = 128
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2
EPOCHS = 1
LEARNING_RATE = 1e-05
MAX_GRAD_NORM = 10
DEVICE = "cuda"
```

## Dataset

The dataset is generously [made available to the public](https://github.com/leslie-huang/UN-named-entity-recognition) by Leslie Huang. It consists of transcribed speeches given at the UN General Assembly from 1993 to 2016, which were scraped from the UN website, parsed (e.g. from PDF), and cleaned. More than 50,000 tokens were manually annotated for NER tags.

``` Python
def clean_tags(tags, tags_to_remove):
    clean_list = []
    for tag in list(tags):
        if tag != "O":
            if tag not in tags_to_remove:
                clean_list.append(tag)
            else:
                clean_list.append("O")
        else:
            clean_list.append("O")
    return clean_list

@dataset("un_ner_dataset")
@resources(path="./UN-named-entity-recognition")
def create_dataset():
    import os
    import itertools
    import pandas as pd
    from collections import Counter
    
    directories = [
        "./UN-named-entity-recognition/tagged-training/",
        "./UN-named-entity-recognition/tagged-test/",
    ]
    data_files = []
    for dir in directories:
        for filename in os.listdir(dir):
            file_path = os.path.join(dir, filename)

            with open(file_path, "r", encoding="utf8") as f:
                lines = f.readlines()
                split_list = [list(y) for x, y in itertools.groupby(lines, lambda z: z == "\n") if not x]
                tokens = [[x.split("\t")[0] for x in y] for y in split_list]
                entities = [[x.split("\t")[1][:-1] for x in y] for y in split_list]
                data_files.append(pd.DataFrame({"tokens": tokens, "ner_tags": entities}))

    dataset = pd.concat(data_files).reset_index().drop("index", axis=1)

    # Cleaning and removing bad tags
    pre_cleanup_tag_counter = Counter([tag for tags in dataset["ner_tags"] for tag in tags])
    tags_to_remove = ["I-PRG", "I-I-MISC", "I-OR", "VMISC", "I-", "0"]
    dataset["ner_tags"] = dataset["ner_tags"].apply(lambda x: clean_tags(x, tags_to_remove))
    tag_counter = Counter([tag for tags in dataset["ner_tags"] for tag in tags])
    dataset_description = """The corpus consists of a sample of transcribed speeches given at the UN General Assembly 
    from 1993-2016, which were scraped from the UN website, parsed (e.g. from PDF), and cleaned. More than 50,000 tokens 
    in the test data were manually tagged for Named Entity Recognition (O - Not a Named Entity; I-PER - Person; I-ORG - 
    Organization; I-LOC - Location; I-MISC - Other Named Entity)."""
    layer.log({"# Examples": len(dataset)})
    layer.log({"Dataset Description": dataset_description})
    layer.log({"Source": "https://github.com/leslie-huang/UN-named-entity-recognition"})
    layer.log({"Raw Tags Counter": pre_cleanup_tag_counter})
    layer.log({"Clean Tags Counter": tag_counter})

    return dataset

ner_dataset = create_dataset()
```

https://app.layer.ai/kaankarakeben/united-nations-ner-finetuning/datasets/un_ner_dataset

## Model: Fine-tuning Pretrained BERT with PyTorch

As stated earlier we will use transfer learning to create our NER model. The pre-trained model we'll use is BERT which large neural network trained on masked language modeling and next sentence prediction tasks. If you are interested in having a deeper understanding, have a look at the [original paper](https://arxiv.org/abs/1810.04805) and this brilliant [blog post](http://jalammar.github.io/illustrated-bert/) by Jay Alammar. The fine-tunning will be a supervised learning effort with our annotated dataset.

We will work [HuggingFace](https://huggingface.co/)'s powerful [transformers](https://github.com/huggingface/transformers) library to get the [PyTorch](https://pytorch.org/) implementation of the pre-trained model as well as the tokenizer object that is required to turn our dataset into the input format for BERT. Below is the code to load the tokenizer and store it on our Layer project.

### Tokenizer

```Python
@pip_requirements(packages=["transformers"])
@fabric("f-medium")
@model(name="bert-base-uncased-tokenizer")
def download_tokenizer():
    from transformers import BertTokenizerFast

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    return tokenizer

tokenizer = download_tokenizer()
```

https://app.layer.ai/kaankarakeben/united-nations-ner-finetuning/models/bert-base-uncased-tokenizer

### Model Inputs

```Python
class PytorchDataset(Dataset):
    def __init__(self, dataframe, tokenizer, tag_to_id, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.tag_to_id = tag_to_id

    def __getitem__(self, index):

        label_all_tokens = True
        tokenized_inputs = self.tokenizer(
            [list(self.data.tokens[index])],
            truncation=True,
            is_split_into_words=True,
            max_length=128,
            padding="max_length",
        )

        labels = []
        for i, label in enumerate([list(self.data.ner_tags[index])]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif label[word_idx] == "0":
                    label_ids.append(0)
                elif word_idx != previous_word_idx:
                    label_ids.append(self.tag_to_id[label[word_idx]])
                else:
                    label_ids.append(self.tag_to_id[label[word_idx]] if label_all_tokens else -100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels

        single_tokenized_input = {}
        for k, v in tokenized_inputs.items():
            single_tokenized_input[k] = torch.as_tensor(v[0])

        return single_tokenized_input

    def __len__(self):
        return self.len
    
def create_model_inputs(dataset, tag_to_id):

    train_dataset = dataset.sample(frac=TRAIN_EXAMPLES_RATIO, random_state=200)
    test_dataset = dataset.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    print("FULL Dataset: {}".format(dataset.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    train = PytorchDataset(train_dataset, tokenizer, tag_to_id, MAX_LEN)
    test = PytorchDataset(test_dataset, tokenizer, tag_to_id, MAX_LEN)

    return train, test

tag_counter = Counter([tag for tags in ner_dataset["ner_tags"] for tag in tags])
tag_to_id = {tag: ix for ix, tag in enumerate(tag_counter.keys())}
train_set, test_set = create_model_inputs(ner_dataset.head(100), tag_to_id)
```

### Training and Evaluation

At this step, we are fine-tuning the model by training the model with pre-trained weights. The method will save the model object at Layer as well as logging the intermediate training loss and the final evaluation results.

```Python
def train(train_set):
    from sklearn.metrics import accuracy_score
    from transformers import BertForTokenClassification
    from torch.utils.data import DataLoader

    train_params = {"batch_size": TRAIN_BATCH_SIZE, "shuffle": True, "num_workers": 0}
    training_loader = DataLoader(train_set, **train_params)

    model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(tag_to_id))
    model.to(DEVICE)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        print(f"Training epoch: {epoch + 1}")
        tr_loss, tr_accuracy = 0, 0
        nb_tr_examples, nb_tr_steps = 0, 0
        tr_preds, tr_labels = [], []

        model.train()  # model in training mode

        for idx, batch in enumerate(training_loader):

            ids = batch["input_ids"].to(DEVICE, dtype=torch.long)
            mask = batch["attention_mask"].to(DEVICE, dtype=torch.long)
            labels = batch["labels"].to(DEVICE, dtype=torch.long)

            outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
            loss = outputs[0]
            tr_logits = outputs[1]
            tr_loss += loss.item()

            nb_tr_steps += 1
            nb_tr_examples += labels.size(0)

            if idx % 100 == 0:
                loss_step = tr_loss / nb_tr_steps
                print(f"Training loss per 100 training steps: {loss_step}")

            # compute training accuracy
            flattened_targets = labels.view(-1)
            active_logits = tr_logits.view(-1, model.num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1)

            # only compute accuracy at active labels
            active_accuracy = labels.view(-1) != -100
            labels = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)

            tr_labels.extend(labels)
            tr_preds.extend(predictions)

            tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
            tr_accuracy += tmp_tr_accuracy

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=MAX_GRAD_NORM)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss = tr_loss / nb_tr_steps
        tr_accuracy = tr_accuracy / nb_tr_steps
        print(f"Training loss epoch: {epoch_loss}")
        print(f"Training accuracy epoch: {tr_accuracy}")

    return model


def evaluate(model, test_set, tag_to_id):
    from sklearn.metrics import classification_report
    from sklearn.metrics import accuracy_score
    
    id_to_tag = {ix: tag for tag, ix in tag_to_id.items()}
    test_params = {"batch_size": VALID_BATCH_SIZE, "shuffle": True, "num_workers": 0}
    testing_loader = DataLoader(test_set, **test_params)

    model.eval()  # model in evaluation mode

    eval_loss, eval_accuracy = 0, 0
    nb_eval_examples, nb_eval_steps = 0, 0
    eval_preds, eval_labels = [], []
    with torch.no_grad():
        for idx, batch in enumerate(testing_loader):

            ids = batch["input_ids"].to(DEVICE, dtype=torch.long)
            mask = batch["attention_mask"].to(DEVICE, dtype=torch.long)
            labels = batch["labels"].to(DEVICE, dtype=torch.long)

            outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
            loss = outputs[0]
            eval_logits = outputs[1]

            eval_loss += loss.item()

            nb_eval_steps += 1
            nb_eval_examples += labels.size(0)

            if idx % 100 == 0:
                loss_step = eval_loss / nb_eval_steps
                print(f"Validation loss per 100 evaluation steps: {loss_step}")

            # compute evaluation accuracy
            flattened_targets = labels.view(-1)
            active_logits = eval_logits.view(-1, model.num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1)

            # only compute accuracy at active labels
            active_accuracy = labels.view(-1) != -100

            labels = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)

            eval_labels.extend(labels)
            eval_preds.extend(predictions)

            tmp_eval_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
            eval_accuracy += tmp_eval_accuracy

    labels = [id_to_tag[id.item()] for id in eval_labels]
    predictions = [id_to_tag[id.item()] for id in eval_preds]

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_steps
    layer.log({"Test Loss": eval_loss, "Test Accuracy": eval_accuracy})

    print(f"Validation Loss: {eval_loss}")
    print(f"Validation Accuracy: {eval_accuracy}")

    print(classification_report(labels, predictions))
    layer.log(classification_report(labels, predictions, output_dict=True))


@pip_requirements(packages=["transformers", "sklearn", "torch"])
@fabric("f-gpu-small")
@model("un_ner_fine-tuned_bert")
def run_model_training():
    model = train(train_set)
    evaluate(model, test_set, tag_to_id)
    return model

model = run_model_training()
```

https://app.layer.ai/kaankarakeben/united-nations-ner-finetuning/models/un_ner_fine-tuned_bert

## Evaluation

Looking at the test set, we are able to achieve an accuracy of 98% and an F1 score of 89% with our trained model. We are pretty accurate with detecting PERSON entities but having low recall with MISCELLANEOUS compared to others. Overall impressive results with a relatively small amount of annotated data!

``` Plain Text

              precision    recall  f1-score   support

       I-LOC       0.91      0.97      0.94       780
      I-MISC       0.80      0.66      0.72       603
       I-ORG       0.80      0.89      0.84       748
       I-PER       0.96      0.97      0.96       178
           O       0.99      0.99      0.99     29144

    accuracy                           0.98     31453
   macro avg       0.89      0.90      0.89     31453
weighted avg       0.98      0.98      0.98     31453

```

##  Conclusion

Extracting named entities from text has many uses that transform the way we interact with these documents. Usage of pre-trained models like BERT and libraries such as Huggingface and PyTorch makes it easy for us to fine-tune general-purpose models into specialist ones. However, for a data scientist life doesn't end with the trained model in a notebook. Features we have shown from Layer allow us to follow the best MLOps practices in building, tracking, and logging all of our artifacts. When all these technologies combine, long-lasting value is unlocked.

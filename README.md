# Fine Tuning BERT For Named Entity Recognition On United Nations Documents

[![Open in Layer](https://development.layer.co/assets/badge.svg)](https://app.layer.ai/kaankarakeben/united-nations-ner-finetuning)
[![Layer NER](https://badgen.net/badge/icon/github?icon=github&label)](https://github.com/kaankarakeben/layer_ner)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zkmCzzdOt4suAEPdQqN6uIWIHXTP-hNn?)

In this tutorial, I will use a fine-tuned BERT model on United Nations meeting documents to detect named entities in a given text.

## How to use

``` Python
!pip install layer --upgrade -qqq
!pip install -U ipython

import layer
import torch
from collections import Counter

layer.login()
layer.init("united-nations-ner-finetuning")

MAX_LEN = 128

ner_dataset = layer.get_dataset("kaankarakeben/united-nations-ner-finetuning/datasets/un_ner_dataset:1.4").to_pandas()
tokenizer = layer.get_model("kaankarakeben/united-nations-ner-finetuning/models/bert-base-uncased-tokenizer:1.3").get_train()
model = layer.get_model("kaankarakeben/united-nations-ner-finetuning/models/un_ner_fine-tuned_bert:1.14").get_train()

tag_counter = Counter([tag for tags in ner_dataset["ner_tags"] for tag in tags])
tag_to_id = {tag: ix for ix, tag in enumerate(tag_counter.keys())}

def predict_ner_example(sentence):
    inputs = tokenizer(
        sentence.split(),
        is_split_into_words=True,
        return_offsets_mapping=True,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )

    id_to_tag = {ix: tag for tag, ix in tag_to_id.items()}
    
    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    # forward pass
    outputs = model(ids, attention_mask=mask)
    logits = outputs[0]

    active_logits = logits.view(-1, model.num_labels)  # shape (batch_size * seq_len, num_labels)
    flattened_predictions = torch.argmax(
        active_logits, axis=1
    )  # shape (batch_size*seq_len,) - predictions at the token level

    tokens = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
    token_predictions = [id_to_tag[i] for i in flattened_predictions.cpu().numpy()]
    wp_preds = list(zip(tokens, token_predictions))  # list of tuples. Each tuple = (wordpiece, prediction)

    prediction = []
    for token_pred, mapping in zip(wp_preds, inputs["offset_mapping"].squeeze().tolist()):
        # only predictions on first word pieces are important
        if mapping[0] == 0 and mapping[1] != 0:
            prediction.append(token_pred[1])
        else:
            continue
            
    return sentence, prediction

sentence = """Expressing deep concern about the impact of the food security crisis on the
assistance provided by United Nations humanitarian agencies, in particular the World
Food Programme."""

sentence, prediction = predict_ner_example(sentence)
print(sentence.split())
print(prediction)

>>> ['Expressing', 'deep', 'concern', 'about', 'the', 'impact', 'of', 'the', 'food', 'security', 'crisis', 'on', 'the', 'assistance', 'provided', 'by', 'United', 'Nations', 'humanitarian', 'agencies,', 'in', 'particular', 'the', 'World', 'Food', 'Programme.']
['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'I-ORG', 'I-ORG', 'I-LOC']
```

## Dataset

The dataset is generously [made available to the public](https://github.com/leslie-huang/UN-named-entity-recognition) by Leslie Huang. It consists of transcribed speeches given at the UN General Assembly from 1993 to 2016, which were scraped from the UN website, parsed (e.g. from PDF), and cleaned. More than 50,000 tokens were manually annotated for NER tags.

https://app.layer.ai/kaankarakeben/united-nations-ner-finetuning/datasets/un_ner_dataset

## Model: Fine-tuning Pretrained BERT with PyTorch

As stated earlier we will use transfer learning to create our NER model. The pre-trained model we'll use is BERT which large neural network trained on masked language modeling and next sentence prediction tasks. If you are interested in having a deeper understanding, have a look at the [original paper](https://arxiv.org/abs/1810.04805) and this brilliant [blog post](http://jalammar.github.io/illustrated-bert/) by Jay Alammar. The fine-tunning will be a supervised learning effort with our annotated dataset.

We will work [HuggingFace](https://huggingface.co/)'s powerful [transformers](https://github.com/huggingface/transformers) library to get the [PyTorch](https://pytorch.org/) implementation of the pre-trained model as well as the tokenizer object that is required to turn our dataset into the input format for BERT. Below is the code to load the tokenizer and store it on our Layer project.

https://app.layer.ai/kaankarakeben/united-nations-ner-finetuning/models/bert-base-uncased-tokenizer
https://app.layer.ai/kaankarakeben/united-nations-ner-finetuning/models/un_ner_fine-tuned_bert

```
@misc{leslie-huang,
  author = {Huang, Leslie},
  title = {UN-named-entity-recognition},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/leslie-huang/UN-named-entity-recognition}},
  commit = {5cb5c5fbacc607542a900d9eff016f9ce4647c9c}
}
```
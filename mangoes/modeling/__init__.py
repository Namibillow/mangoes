# -*- coding: utf-8 -*-
"""
The modeling subpackage provides functionality for using BERT models for various tasks, including feature extraction,
pretraining, and a number of fine-tuning tasks. It is a interface into the Huggingface transformers and tokenizers
libraries (https://huggingface.co/transformers/ and https://huggingface.co/docs/tokenizers/python/latest/), meant to
provide an easy-to-use API for using BERT for contextual embeddings or training/inference.

The main component of the subpackage are the task classes. Each possible task is implemented as a class that inherits
from :class:`mangoes.modeling.bert_base.TransformerModel`. This API provides full pipeline functionality for each task,
from raw textual inputs to inference, and training, saving, and loading. Each class has a tokenizer and model as
attributes.

The API for these task classes include 5 main functions:
    - load():
        This function will load a pretrained tokenizer and model. Model loading includes loading a pretrained base
        BERT model (while initializing possible fine-tuning head(s)) for use in fine-tuning, or loading a pretrained base
        BERT model and trained classification heads. As such, for all tasks except pretraining from scratch, this is the
        preferred way to instantiate a task class.
        Arguments into this load method can be paths to local directories containing pretrained tokenizers/models, or
        keyword identifiers that map into Huggingface's model hub (see https://huggingface.co/models for more info).

    - save():
        This function is a way to save a trained model and/or tokenizer to a local directory.

    - train():
        The train function provides an easy-to-use interface for training the model for the associated task.
        Under the hood, this function makes use of the transformers.Trainer interface (see
        https://huggingface.co/transformers/training.html#trainer). The train methods in the task classes are meant to have
        intuitive default functionality, while providing users powerful options for customization. With this in mind, there
        are 3 ways to call each task class' train method:

        - raw data:
            Provide raw (ie, not tokenized or featurized) data as input to the method. Dataset classes will be
            instantiated internally using the mangoes default task Dataset classes in
            :module:`mangoes.modeling.training_utils`, which will then be used to instantiated a transformers.Trainer to
            train the model.

        - dataset objects:
            Provide your own dataset objects as input, instead of the raw data. These objects should
            subclass the torch.Dataset (https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset), as this is
            what is used internally in the Trainer class. A Trainer object will be instantiated, using these dataset
            classes, inside the train method. This option exists if users want dataset functionality that is different
            than the provided default dataset classes.

        - Trainer object:
            Provide an instantiated transformers.Trainer object as input to the train class. Providing the most
            customization possible, this option exists for users who want special functionality while training (ie, by
            subclassing the transformers.Trainer class and overriding methods).

        The first two options allow the user to pass keyword arguments for training hyper-parameters (see the API
        docs for specifics), while the third allows users to instantiated their own Trainer object with what ever
        hyper-parameters they want. See examples in the use-cases sections of the documentation, or in the BERT demo
        notebooks.

    - predict():
        The predict methods provide a straightforward inference call for each task class, taking as input
        non-tokenized text for the associated task and directly outputting the model's predictions.

    - generate_outputs():
        This method is also a way to use the model for inference. However, it differs from predict() in that the outputs
        are not processed to parse a direct prediction (for example by calling argmax on output probabilities).
        Also, generate_outputs can return hidden state outputs and attention matrices for each layer in
        the base BERT model, in addition to possible fine tuning head layer outputs. Furthermore, information on how the
        input text was tokenized is returned as well. This method is useful when users want to analyze the
        outputs/network more than just getting a direct prediction. For more information, see the API docs.

In addition to these 5 methods, the task classes all expose their tokenizers and models as attributes, giving direct
access to the Huggingface tokenizer and model classes, if needed.
"""
from mangoes.modeling.bert_base import *
from mangoes.modeling.bert_finetuning import *
from mangoes.modeling.bert_pretraining import *
from mangoes.modeling.coref import *
from mangoes.modeling.tokenizer import *
from mangoes.modeling.training_utils import *

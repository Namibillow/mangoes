======================
Use cases: Contextual Language Models
======================


Some miscellaneous options and use cases for using Mangoes to work with contextual language models.

.. note:: **Doctest Mode**

   The code in the above examples are written in a *python-console* format.
   If you wish to easily execute these examples in **IPython**, use::

      %doctest_mode

   in the IPython console. You can then simply copy and paste the examples
   directly into IPython without having to worry about removing the **>>>**
   manually.


Generate embeddings from a pretrained BERT model
----------------------------------------------------------------------------

If you have text that you want to generate embeddings for from any Transformer model, you can use the :func:`mangoes.modeling.PretrainedTransformerModelForFeatureExtraction.load()` method to instantiate a pre-trained model.
Make sure to use the right pre-trained tokenizer for the pre-trained model. For downloading weights, see https://huggingface.co/transformers/pretrained_models.html for official models and https://huggingface.co/models for user uploaded models.

    >>> from mangoes.modeling import PretrainedTransformerModelForFeatureExtraction
    >>>
    >>> # downloads pretrained bert_base_uncased tokenizer (first argument) and model weights (second argument)
    >>> # If the device argument is None, it will use GPU if it's available, else cpu
    >>> model = PretrainedTransformerModelForFeatureExtraction.load("bert-base-uncased", "bert-base-uncased", device=None)
    >>> # alternatively, you can download the tokenizer and weights of a user-uploaded model, in this case Spanbert
    >>> model = PretrainedTransformerModelForFeatureExtraction.load("SpanBERT/spanbert-base-cased", "SpanBERT/spanbert-base-cased",  device=None)

Calling :func:`predict()` on the model and passing in text will return the hidden state of the last transformer layer for each token

    >>> text = ["I'm a test sentence.", "This is another test sentence"]
    >>> outputs = model.predict(text)
    >>> print(len(outputs))           # one list of hidden layer outputs per input sentences
    2
    >>> print(len(outputs[0]))        # sequence length
    9
    >>> print(len(outputs[0][-1]))    # size of hidden state of last layer
    768

Alternatively, you can use the :func:`generate_outputs()` function to optionally get all layers' hidden states, attention matrices, as well as the sub-word token mappings.

    >>> outputs = model.generate_outputs(text, pre_tokenized=False, output_hidden_states=True, output_attentions=False, word_embeddings=False)
    >>> print(outputs.keys())
    dict_keys(['last_hidden_state', 'pooler_output', 'hidden_states', 'attentions', 'offset_mappings'])
    >>> print(outputs["hidden_states"][-1].shape)  # shape of last layers hidden state: batch_size, max_sequence_length, hidden_size
    >>> print(outputs["attentions"][-1].shape)     # shape of last layers' attention matrix batch_size, num_attention_heads, max_sequence_length, max_sequence_length
    >>> print(outputs["offset_mappings"].shape)    # sub-token mapping shape: batch_size, max_sequence_length, 2:(start and end indices)

You can also pass in sentences that have already been split on whitespace.

    >>> input_text = "This is a test sentence".split()
    >>> print(input_text)
    ['This', 'is', 'a', 'test', 'sentence']
    >>> outputs = model.generate_outputs(input_text, pre_tokenized=True, output_hidden_states=True, output_attentions=False, word_embeddings=False)
    >>> print(outputs["hidden_states"][-1].shape)
    torch.Size([1, 7, 768])

Since BERT uses the Wordpiece sub-word tokenizer, sometimes words are split into subwords. You can obtain the subword tokens by using the "offset_mappings" value in the output dictionary of the generate_outputs method.
Or, you could directly call the tokenizer, if you only want the tokenization and not the model inference.

    >>> input_text = "The word decreasingly is two tokens"
    >>> outputs = model.generate_outputs(input_text, pre_tokenized=False, output_hidden_states=True, output_attentions=False, word_embeddings=False)
    >>> print(outputs["hidden_states"][-1].shape)
    torch.Size([1, 10, 768])
    >>> print([input_text[token[0]:token[1]] for token in outputs["offset_mapping"]])  # print the subword tokens using the generate outputs output
    ['', 'The', 'word', 'decreasing', 'ly', 'is', 'two', 'token', 's', '']  # 8 subwords + start and end special tokens = 10 tokens total
    >>> tokenizer_output = model.tokenizer(input_text, return_offsets_mapping=True)             # tokenize the input
    >>> print([input_text[token[0]:token[1]] for token in tokenizer_output["offset_mapping"]])  # print the subword tokens using the tokenizer output
    ['', 'The', 'word', 'decreasing', 'ly', 'is', 'two', 'token', 's', '']

If you would like to average subword embeddings back to word embeddings (as well as strip special tokens), turn the `word_embeddings` flag to True.
If the input is pre-tokenized, this functionality will average sub-word tokens together such that each individual token in the pre-tokenized input has one embedding.
If the input is not pre-tokenized, setting word_embeddings to True will split the text on whitespace and consolidate sub-word token embeddings into word embeddings based on the white space split.

    >>> outputs = model.generate_outputs(input_text, pre_tokenized=False, output_hidden_states=True, output_attentions=False, word_embeddings=True)
    >>> print(outputs["hidden_states"][-1].shape)
    torch.Size([1, 6, 768]) # shape is (batch size or num sentences, num words, embedding size)


Pre-train a BERT model on a corpus of text
---------------------------------------------
If you would like to train a BERT model from scratch on a corpus, this is possible with the following steps:
First, instantiate a new Wordpiece tokenizer, train it on the corpus, and save it to a directory to be loaded while instantiating the pretraining class.
Alternatively, you can use a pretrained tokenizer and pass it's identifier as an argument to the pretraining task class instantiation.

    >>> from mangoes.modeling import BERTWordPieceTokenizer
    >>> corpus_path = "./mangoes/resources/en/wikipedia_en_2013.100K.tokenized.txt" # could also list of file paths
    >>> tokenizer = BERTWordPieceTokenizer(vocab=None)
    >>> tokenizer.train(corpus_path, vocab_size=20000, min_frequency=2)
    >>> print(tokenizer.get_vocab_size())
    30000
    >>> tokenizer.save("./tokenizer/")    # saves the tokenizer vocab, as well as configuration

Next, instantiate a new BERT model class with the tokenizer argument as the path to the saved tokenizer file. Note that BERT models can be trained using only the masked language model (MLM) task,
or using the MSM *and* Next Sentence Prediction (NSP) task, like in the original paper. Use the :class:`mangoes.modeling.BERTForMaskedLanguageModeling` class to pretrain using only the MLM task,
and use the :class:`mangoes.modeling.BERTForPreTraining` class to pretrain using both the MSM and NSP tasks.
Here we will use the MSM only pretraining, as recent papers have found the NSP task to not be important to achieving good performance.

    >>> from mangoes.modeling import BERTForMaskedLanguageModeling
    >>> model = BERTForMaskedLanguageModeling("./tokenizer/", num_hidden_layers=6) # pass in the directory containing the saved tokenizer, as well as any BERT architecture keyword arguments
    >>> print(model.tokenizer.vocab_size)
    20000

Finally, we train the model on the corpus:

    >>> model.train(output_dir="./testing/", train_text=corpus_path, max_len=512, num_train_epochs=3, learning_rate=0.0001, dataloader_num_workers=4)

For increased customization, you can instantiate and pass your own Transformers.DataCollator, torch.utils.data.Dataset, or even Transformers.Trainer.
Below is an example of passing in Dataset objects instead of the raw corpus path. The :class:`mangoes.modeling.MangoesLineByLineDataset` is what is used internally in the train function,
but any subclass of torch.utils.data.Dataset can be passed in as the data arguments. Here we will additionally pass in a French Wikipedia argument as the validation dataset:

    >>> from mangoes.modeling import MangoesLineByLineDataset
    >>>
    >>> eval_corpus_path = "./data/wiki_article_fr"
    >>> train_dataset = MangoesLineByLineDataset(corpus_path, model.tokenizer, max_len=256)
    >>> eval_dataset = MangoesLineByLineDataset(eval_corpus_path, model.tokenizer, max_len=256)
    >>>
    >>> model.train(train_dataset=train_dataset, eval_dataset=eval_dataset, output_dir=model_output_dir,
    >>>           num_train_epochs=4, learning_rate=0.00005, logging_steps=40, evaluation_strategy="epoch")

Once the model is done training, you can use it to predict masked tokens using the :func:`.predict()` function:

    >>> print(model.predict(f"I {model.tokenizer.mask_token} getting up early", top_k=1))
    >>> [{'sequence': '[CLS] i was getting up early [SEP]', 'score': 0.6499782204627991, 'token': 2001, 'token_str': 'was'}]

Alternatively, you could use the :func:`generate_outputs()` to get a more detailed output, such as the pre-softmax scores for every token in the vocabulary, for every token in the input sequence:

    >>> outputs = model.generate_outputs(input_text, output_hidden_states=True, output_attentions=True)
    >>> # outputs["logits"] contains the mlm scores, with shape (1, seq_length, vocab_size)
    >>> # to get the score for a particular word, use the tokenizer to find the index of the word and extract the score
    >>> print(outputs["logits"][0][2][model.tokenizer.convert_tokens_to_ids("the")])    # the masked token is the 3rd token in the sequence (after the start special token, and the "I" token)

Finally, the model (and tokenizer, if needed) can be saved with the :func:`.save()` function.

    >>> model.save("path/to/output/directory/", save_tokenizer=False)


Fine-tuning a BERT model for token or sequence classification
-------------------------------------------------------------

You can fine-tune a pretrained model for sequence classification (i.e. sentiment analysis) or token classification (i.e. POSs tagging) using the
:class:`mangoes.modeling.BERTForSequenceClassification` or :class:`mangoes.modeling.BERTForTokenClassification` classes.
Here's an example of sentiment analysis using the nlp library's interface to the imdb dataset.
First, we prepare the dataset and get it into the format needed:

    >>> from mangoes.modeling import BERTForSequenceClassification
    >>> from nlp import load_dataset
    >>>
    >>> train_dataset, test_dataset = load_dataset('imdb', split=['train', 'test'])
    >>> train_texts = [x['text'] for x in train_dataset]
    >>> train_targets = [x['label'] for x in train_dataset]
    >>> test_texts = [x['text'] for x in test_dataset]
    >>> test_targets = [x['label'] for x in test_dataset]

Next, we instantiate and train the model, passing in the raw (ie, not tokenized or tensorized) data to the train argument.
The model below is instantiated used and pretrained BERT base model from the Huggingface servers. Users can also pass in the directory where they have saved a pretrained base model.
Sometimes, users would like calculate metrics while training to monitor. This can be done by defining a metrics function and passing it
to the train method using the 'compute_metrics' keyword. For more information, see https://huggingface.co/transformers/training.html#trainer:

    >>> from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    >>>
    >>> def compute_metrics(pred):
    >>>     labels = pred.label_ids
    >>>     preds = pred.predictions.argmax(-1)
    >>>     precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    >>>     acc = accuracy_score(labels, preds)
    >>>     return {
    >>>         'accuracy': acc,
    >>>         'f1': f1,
    >>>         'precision': precision,
    >>>         'recall': recall
    >>>     }
    >>>
    >>> model = BERTForSequenceClassification.load("bert-base-uncased", "bert-base-uncased", labels=["pos, neg"],
    >>>                                             label2id={'neg': 0, 'pos': 1})
    >>> model.train(train_text=train_texts, train_targets=train_targets,
    >>>                eval_text=test_texts, eval_targets=test_targets, evaluation_strategy="epoch",
    >>>                output_dir="./testing/", max_len=512, num_train_epochs=3, compute_metrics=compute_metrics)

Alternatively, you could instantiate your own torch.utils.data.Dataset subclass and pass this in as well.

    >>> train_dataset = MangoesTextClassificationDataset(train_texts, train_targets, model.tokenizer, max_len=512, label2id={'neg': 0, 'pos': 1})
    >>> eval_dataset = MangoesTextClassificationDataset(test_texts, test_targets, model.tokenizer, max_len=512, label2id={'neg': 0, 'pos': 1})
    >>> model.train(train_dataset=train_dataset, eval_dataset=eval_dataset, evaluation_strategy="epoch",
    >>>                output_dir="./testing/", max_len=512, num_train_epochs=1,
    >>>                per_device_train_batch_size=4)

Once the model has been fine-tuned, it can be used for inference using the predict or generate_outputs methods.

    >>> predictions = loaded_model.predict("This is a good movie")
    >>> print(predictions)
    [{'label': 'pos', 'score': 0.9922362565994263}]
    >>> outputs = loaded_model.generate_outputs("This is a good movie", output_hidden_states=True, output_attentions=True)
    >>> print(outputs.keys())
    dict_keys(['logits', 'hidden_states', 'attentions', 'offset_mappings'])


Fine-tuning a BERT model for Question Answering
-------------------------------------------------------------

You can fine tune a pretrained BERT model for question answering using the :class:`mangoes.modeling.BERTForQuestionAnswering` class.
An example using a toy dataset:

    >>> from mangoes.modeling import BERTForQuestionAnswering
    >>> # first we load a pretrained base model
    >>> pretrained_mod = BERTForQuestionAnswering.load("bert-base-uncased", "bert-base-uncased")

Here's a toy dataset we can use. A question answering dataset includes the questions, contexts, answers, and the indices at which the answers start in the context strings.

    >>> QUESTIONS = ["What is the context for this question?", "What kind of question is this?"]
    >>> CONTEXTS = ["This is context for the test questions.", "This is context for the test question."]
    >>> ANSWERS = ["This", "test"]
    >>> ANSWER_START_INDICES = [CONTEXTS[i].find(ANSWERS[i]) for i in range(len(ANSWERS))]

Next, we can fine tune the model:

    >>> pretrained_mod.train(train_question_texts=QUESTIONS, train_context_texts=CONTEXTS, train_answer_texts=ANSWERS,
    >>>                         train_start_indices=ANSWER_START_INDICES, output_dir="./output_dir/", num_train_epochs=3)

Alternatively, we can instantiate our own transformers.Trainer object and pass this in.
Notice the "freeze_base" flag, which will freeze the base layers during training so only the task heads get updated:

    >>> from transformers import Trainer, TrainingArguments, PrinterCallback
    >>> from mangoes.modeling import MangoesQuestionAnsweringDataset
    >>>
    >>> train_dataset = MangoesQuestionAnsweringDataset(pretrained_mod.tokenizer, train_questions,
    >>>                                                 train_contexts, train_answers, train_starts)
    >>>
    >>> train_args = TrainingArguments(output_dir="./model_ckpts/", num_train_epochs=1, learning_rate=0.00005,
    >>>                                             per_device_train_batch_size=32, logging_steps=4)
    >>> trainer = Trainer(pretrained_mod.model, args=train_args, train_dataset=train_dataset,
    >>>                   pretrained_mod=loaded_model.tokenizer, callbacks=[PrinterCallback])
    >>>
    >>> pretrained_mod.train(trainer=trainer, freeze_base=True)

Once the model is trained, we can predict answers using the :func:`.predict()` function:

    >>> predictions = pretrained_mod.predict(question=QUESTIONS[0], context=CONTEXTS[0])
    >>> print(predictions["answer"])    # print the predicted answer text

Alternatively, you could use :func:`.generate_outputs()` to get more detailed output, such as the start and end logits of the answer, and the hidden_states and attention matrices.

    >>> outputs = pretrained_mod.generate_outputs(question=QUESTIONS, context=context, pre_tokenized=False, output_hidden_states=True, output_attentions=True)
    >>> print(outputs["start_logits"])


Fine-tuning a BERT model for Multiple Choice Questions
-------------------------------------------------------------

Another fine-tuning task is training a model to answer multiple choice questions. We'll start by loading a pretrained base model:

    >>> loaded_model = BERTForMultipleChoice.load("bert-base-cased", "SpanBERT/spanbert-base-cased")

We'll use a subset of the hellaswag (extension of SWAG) dataset for training:

    >>> from nlp import load_dataset
    >>>
    >>> train_dataset, eval_dataset = load_dataset('hellaswag', split=['train', 'validation'])
    >>> train_contexts = [x['ctx_a'] for x in train_dataset][:65]
    >>> train_choices = [[x['ctx_b'] + " " + ending for ending in x['endings']] for x in train_dataset][:65]
    >>> train_labels = [x['label'] for x in train_dataset][:65]
    >>> eval_contexts = [x['ctx_a'] for x in eval_dataset][:100]
    >>> eval_choices = [[x['ctx_b'] + " " + ending for ending in x['endings']] for x in eval_dataset][:100]
    >>> eval_labels = [x['label'] for x in eval_dataset][:100]

Next, we can pass this raw data into the train function along with any training parameters.
One notable hyperparameter is the "task_learn_rate", which is the learning rate for the parameters in the task head layers.
The base BERT model parameters will use the "learning_rate" learning rate.
Alternatively, users can set the "freeze_base" parameter to True, and the base BERT layers will be frozen and not updated during training.

    >>> loaded_model.train(train_question_texts=train_contexts, eval_question_texts=eval_contexts,
    >>>                 train_choices_texts=train_choices, eval_choices_texts=eval_choices,
    >>>                 train_labels=train_labels, eval_labels=eval_labels, learning_rate=0.0005,
    >>>                 per_device_train_batch_size=8, per_device_eval_batch_size=8, logging_steps=4,
    >>>                 max_len=384, output_dir="./model_ckpts/", num_train_epochs=1, task_learn_rate=0.005)

We can then use the predict or generate_outputs functions to use the model for inference:

    >>> questions = "What did the cat say to the dog?"
    >>> choices = ["It said meow", "it said bark"]
    >>>
    >>>
    >>> predictions = loaded_model.predict(questions, choices)
    >>> print(predictions)
    [{'answer_index': 0, 'score': 0.5091835856437683, 'answer_text': 'It said meow'}]
    >>> outputs = loaded_model.generate_outputs(questions, choices)
    >>> print(outputs.keys())
    dict_keys(['logits', 'offset_mappings'])


Fine-tuning a BERT model for Co-reference Resolution
-------------------------------------------------------------

Another fine tuning task is co-reference resolution. One example of a coref dataset is the ONTONOTES dataset.
We can start by initializing a model by passing in the name of a pretrained tokenizer and base BERT model.
If using a dataset that includes metadata (ie speaker and genre information), we set the "use_metadata" flag to true.

    >>> loaded_model = BERTForCoreferenceResolution.load("bert-base-cased", "SpanBERT/spanbert-base-cased", use_metadata=True)

We can load a small ONTONOTES example from json file, which we'll use for fine-tuning the model:

    >>> import json
    >>>
    >>> with open('data/coref_data.json') as json_file:
    >>>     data_dict = json.load(json_file)
    >>> print(data_dict.keys())
    dict_keys(['sentences', 'clusters', 'speakers', 'genres'])

Next, we can pass in the data to the train method:

    >>> loaded_model.train(output_dir="./model_ckpts/", train_documents=data_dict["sentences"],
    >>>                    train_cluster_ids=data_dict["clusters"], train_speaker_ids=data_dict["speakers"],
    >>>                    train_genres=data_dict["genres"],
    >>>                    num_train_epochs=1, learning_rate=0.0005,
    >>>                    logging_steps=2, task_learn_rate=0.001, evaluation_strategy="epoch")

We can then use the model for inference using predict or generate_outputs, taking a random example from the data:

    >>> # pre-tokenized
    >>> document = data_dict["sentences"][50][7:12]
    >>> speakers = data_dict["speakers"][50][7:12]
    >>> genre = data_dict["genres"][50]
    >>>
    >>> # not pre-tokenized
    >>> input_doc = [' '.join(sent) for sent in document]
    >>> input_speaker = [sent[0] for sent in speakers]
    >>>
    >>> predictions = loaded_model.predict(document, pre_tokenized=True, speaker_ids=speakers, genre=genre)
    >>>
    >>> for coref in predictions:
    >>>     print(coref["cluster_tokens"])
    >>>
    >>> outputs = loaded_model.generate_outputs(input_doc, pre_tokenized=False, speaker_ids=input_speaker, genre=genre)
    >>> print(outputs.keys())
    dict_keys(['loss', 'candidate_starts', 'candidate_ends', 'candidate_mention_scores', 'top_span_starts', 'top_span_ends',
        'top_antecedents', 'top_antecedent_scores', 'flattened_ids', 'flattened_text'])


Using a pretrained transformers model other than BERT
-----------------------------------------------------

Mangoes provides a detailed interface for BERT and various BERT fine-tuning functionality, but it is possible to use a non-BERT model that is available from transformers.
For example, one can use a pretrained ALBERT model using the :class:`mangoes.modeling.PretrainedTransformerModel` class:

    >>> from mangoes.modeling import PretrainedTransformerModelForFeatureExtraction
    >>> albert_model = PretrainedTransformerModelForFeatureExtraction.load("albert-base-v1", "albert-base-v1")
    >>> embeddings = albert_model.generate_outputs("This is a test sentence", output_hidden_states=True)["hidden_states"][-1].cpu().numpy()
    >>> print(embeddings.shape)  # (1, 7, 768)


# Question Answering

### Problem description
It is about a question answering task on long greek documents. The length of each document is between 12000 - 22000 characters as the EDA shows.

### Approach
The providing solution is based on [BERT](https://github.com/google-research/bert) and [Transformers](https://github.com/huggingface/transformers). As pretrained model I used the [nlpaueb/bert-base-greek-uncased-v1](https://huggingface.co/nlpaueb/bert-base-greek-uncased-v1) and for the implementation of the TFBertForQuestionAnswering model I followed the instructions under the official documentation of Transformers page. I fine-tuned the model on a GPU with the following parameters:
1. epochs: 2
2. batch_size: 8
3. learning_rate: 5e-5

The above parameters delivered the best results after experiments that I made by tuning those parameters as it is suggested under the [official paper](https://arxiv.org/pdf/1810.04805.pdf).

My approach is written on a Google Colab Notebook. Initially I trained the model on a TPU. However I was not able to use the TPU for the prediction part, therefore I switched to GPU and I am using the GPU for training and prediction too.

### Challenges
Bert is a really powerful model for tackling a question-answering problem. However, it comes up with the limitation of 512 tokens and the documents under my data-set were really long (12000 - 22000 characters). In order to handle this limitation I wrote the function ```truncate_doc_for_answer```, which truncates each document (before the tokenization) and with this approach the truncated document contains the answer, it is ready for tokenization and it boosts the training part. 

### Data set
Due to some confidentiality concerns I am not able to share the data set (training, validation and testing). There are three files that are required: 
1. A json file which contains the documents and the keys are: id, content.
2. A csv file which contains the questions and the keys are: question_id, original_question.
3. A json file which contains the answers for all the questions for each document. For each answer (for each question and for all documents) there are the answer_start_at and answer_end_at indexes. 

### Instructions
I provide a Google Colab Notebook and for reproducing the results you need:
1. Turn on the GPU by navigating to Edit→Notebook Settings and selecting GPU from the Hardware Accelerator drop-down
2. You should provide the model with your own data set. There is a code cell which indicates to add your own data set.
3. Run all the cells of the question-answering-on-long-doc.ipynb file.

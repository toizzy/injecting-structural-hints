Code for the paper "Injecting structural hints: Using language models to study inductive biases in language learning" by Isabel Papadimitriou and Dan Jurafsky

Code for creating synthetic corpora and loading into a huggingface dataset is in `synthetic-corpora`

We pretrained models using Mistral [Mistral](https://github.com/stanford-crfm/mistral/tree/main/conf). We provide sample Mistral configs here in `sample-mistral-configs`. A sample mistral command to pretrain a model on the 1% mixed language is:

`CUDA_VISIBLE_DEVICES=0 python train.py --config conf/mistral-small.yaml --nnodes 1 --nproc_per_node 1 --training_arguments.fp16 true --training_arguments.per_device_train_batch_size 64 --dataset.name mixed-parens0.01_vocab500-zipf --run_id mixed-parens0.01_vocab50K-zipf`

Code for then finetuning on wikitext is in `finetuning`


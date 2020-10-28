# Transformers-bench

Transformers-bench aims at comparing several fast transformer models in an NLP setting.

## Supported models

Models are implemented using Huggingface transformers library (redefining the attention mechanism). 

Available models:
* Roberta (baseline)
* Kernel
* linformer
* Avgpooling
* Maxpooling
* Efficient
* Longformer
* Block
* Reformer

## Tasks and metrics

Models are compared using MLM.

Monitored metrics are bpc/perplexity, CrossEntropy, Accuracy, f1, precision, recall.

## Datasets

Use `data/prepare_data.sh` to download and prepare (generate train, valid, test splits) the datasets.

Supported datasets:
 * enwik8 
 * enwik9 
 * text8 (token-level)
 * wikitext-2 
 * wikitext-103

## Dev Tasks
- Refactoring
    - [x] basic refactoring
    - [ ] avoid loading model configuration twice
    - [ ] use @dataclass for model arguments (uniformize training, dataset and model configuration)
- [ ] Main  
    - [x] Discuss/Fix memory usage
    - [x] Load provided model configuration
    - [ ] Enable loading pretrain model (using checkpoints)  
        - [x] Enable loading a pretrained roberta model for deep initialization
        - [x] Enable loading a model AND its trainer checkpoint 
        - [ ] Add a checkpoint argument to the parser (to be discussed)
        
## Memo
- Evaluation using enwik9 is long (6 minutes)
- Debug
    - (seb@maccpu usage) uncomment pytorch-fast-transformers dependencies exclusion (model.attention, model.building)
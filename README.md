# Transformers-bench

Transformers-bench aims at comparing several fast transformer models in an NLP setting.

# Dependies
* Pytorch >= 1.5.0 (may work with an older version)
* Transformers >= 3.5.0 

### Optional dependencies:
* [pykeops](https://github.com/getkeops/keops) (requires cmake and gcc/g++ == 7.x to be compatible with cuda>=10.0)
* [pytorch-fast-transformer](https://github.com/idiap/fast-transformers) (may be removed in the future)
```
pip install pykeops[devtools]
pip install pytorch-fast-transformers
```

## Supported models

Models are implemented using Huggingface transformers library (redefining the attention mechanism). 

Available models:
* Roberta (baseline)
* Kernel 
* Linformer
* Cosine
* Avgpooling
* Maxpooling
* Efficient
* Longformer
* Block
* Reformer
* Local (block based)
* Local-global (block based)

See model configuration below

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

## Configuration 

### Model configuration 

#### Generic parameters

* `model`: model flag
    * `roberta`: (RobertaConfig, RobertaSelfAttention),
    * `kernel`: (KernelConfig, KernelSelfAttention),
    * `linformer`: (LinformerConfig, LinformerSelfAttention),
    * `avgpooling`: (AvgPoolingConfig, AvgPoolingSelfAttention),
    * `maxpooling`: (MaxPoolingConfig, MaxPoolingSelfAttention),
    * `efficient`: (EfficientConfig, EfficientSelfAttention),
    * `longformer`: (LongformerConfig, LongformerSelfAttention_),
    * `block`: (BlockConfig, BlockSelfAttention),
    * `reformer`: (ReformerConfig, ReformerSelfAttention)
    
* `from_pretrained_roberta` true/false

* `num_hidden_layers`: 12,
* `hidden_size`: 768,
* `intermediate_size`: 3072,
* `num_attention_heads`: 12,

* `max_position_embeddings`: 512,
* `sequence_len`: 512,

* `hidden_dropout_prob`: 0.0,
* `attention_probs_dropout_prob`: 0.0,


#### Model specific parameters 

**Pooling models**

* `kernel` 
* `stride` 

**Linformer**

* `projection_length`
* `projection_bias`: false

**Longformer**

* `attention_window`


**Reformer**

* `chunk_size`
* `bits`
* `rounds`

**Block**

* `chunk_size`


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
    - config Roberta https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-config.json
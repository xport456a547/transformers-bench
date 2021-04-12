# Transformers-bench

Transformers-bench aims at comparing several fast transformer models in an NLP setting.

# Dependencies
* Pytorch >= 1.5.0 (may work with an older version)
* Transformers >= 4.0.0 

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
* Kernel ([see](https://arxiv.org/abs/2006.16236))
* Linformer ([see](https://arxiv.org/abs/2006.04768))
* Cosine
* Avgpooling
* Maxpooling
* Efficient ([see](https://arxiv.org/abs/1812.01243))
* Longformer ([see](https://arxiv.org/abs/2004.05150), based on HF)
* Local (attention window attention, based on HF)
* Block (non overlapping blocks attention)
* Block-Local (overlapping blocks to approximate local attn)
* Block-Global (Block-Local + extended context with high norm tokens)
* Block-Global-Merged (Block-Global with merged computation, compatible with KeOps)
* BigBird ([see](https://arxiv.org/pdf/2007.14062.pdf), based on HF **sparse** implementation)
* LSH (use HF implementation of [Reformer](https://arxiv.org/abs/2001.04451))
* LSH-FT (use pytorch-torch-transformers implementation: optional)

See model configuration below

## Tasks and metrics

Models are compared using MLM.

Monitored metrics are bpc/perplexity, CrossEntropy, Accuracy, f1, precision, recall.

## Datasets

Use `data/prepare_data.sh` to download and prepare (generate train, valid, test splits) the datasets. (requires unzip)

Supported datasets:
 * enwik8 
 * enwik9 
 * text8 (token-level)
 * wikitext-2 
 * wikitext-103

## Configuration 

### Training configuration

1 step = forward pass + backward pass + weights update
For each experiment, we process 2^18 tokens per step.

Example with 2 GPUs, sequence length of 4096 and batch of 2 sequences per GPU:
* tokens per forward pass = 2 * 4096 * 2 = 16384
* accumulation steps = 2^18 / 16384 = 16

### Model configuration 

#### Generic parameters

* `model`: model flag
    * `roberta`: (RobertaConfig, RobertaSelfAttention)
    * `kernel`: (KernelConfig, KernelSelfAttention)
    * `linformer`: (LinformerConfig, LinformerSelfAttention)
    * `avgpooling`: (AvgPoolingConfig, AvgPoolingSelfAttention)
    * `maxpooling`: (MaxPoolingConfig, MaxPoolingSelfAttention)
    * `cosine`: (CosineConfig, CosineSelfAttention)
    * `efficient`: (EfficientConfig, EfficientSelfAttention)
    * `longformer`: (LongformerConfig, LongformerSelfAttention)
    * `local`: (LocalConfig, LocalSelfAttention)
    * `block`: (BlockConfig, BlockSelfAttention)
    * `block-local`: (BlockLocalConfig, BlockLocalSelfAttention)
    * `block-global`: (BlockGlobalConfig, BlockGlobalSelfAttention)
    * `block-global-merged`: (BlockGlobalConfig, BlockGlobalSelfAttentionMerged)
    * `bigbird`: (BigBirdConfig, BigBirdBlockSparseAttention)
    * `lsh`: (LSHConfig, LSHSelfAttention)
    * `lsh-ft`: (LSHFTConfig, LSHFTSelfAttention)
    
    
* `from_pretrained_roberta` true/false

* `num_hidden_layers`: 12,
* `hidden_size`: 768,
* `intermediate_size`: 3072,
* `num_attention_heads`: 12,

* `max_position_embeddings`: 512,
* `sequence_len`: 512, (change for long range dependencies)

* `hidden_dropout_prob`: 0.1,
* `attention_probs_dropout_prob`: 0.1,


#### Model specific parameters 

Models rely on `sequence_len` and  `max_position_embeddings` which can be different.
For example:
* `sequence_len`: 2048
* `max_position_embeddings`: 512
In this case we duplicate: 2048 = concat[512, 512, 512, 512]

**Pooling models**

* `kernel`: *int*, size of kernel
* `stride`: *int*, size of stride

**Linformer**

* `projection_length`: *int*, projection length inside self attention
* `projection_bias`: *bool*, use bias

**Longformer**

* `attention_window`: *int*, window size, same for all layers

**Reformer**

* `chunk_size`: *int*, size of chunks
* `bits`: *int*, number of projections
* `rounds`: *int*, number of rounds

**Block**

* `chunk_size`: *int*, attention window is made of 3 blocks of chunk_size/2

**Block-local**

* `chunk_size`: *int*, attention window is made of 3 blocks of chunk_size/2
* `use_global`: *bool*, add first token global connection

**Block-Global**

* `chunk_size`: *int*, attention window is made of 3 blocks of chunk_size/2
* `topk`: *int*, topk attention window is made of 3 blocks of topk/2
* `factor`: *int*, select top 1/factor % tokens with highest norm
* `circular`: *bool*, attention window connects to the other side
* `keops`: *bool* (merged version only), rely on keops for attention computation, slower but memory efficient


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

# transformers-bench

Use `data/prepare_data.sh` to load the datasets

### Tasks
- [ ] Main  
    - [ ] Discuss/Fix memory usage
    - [ ] Load provided model configuration
    - [ ] Enable loading pretrain model (using checkpoints)  
        - [x] Enable loading a pretrained roberta model for deep initialization
        - [x] Enable loading a model AND its trainer checkpoint 
        - [ ] Add a checkpoint argument to the parser (to be discussed)
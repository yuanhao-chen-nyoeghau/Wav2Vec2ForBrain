# Brain2Text
This project was started in the context of Competitive Programming with Deep Learning Seminar at HPI. 
We attempt to decode thought about text based on brain electrode data

## Update conda environment from environment.yaml
```
conda activate b2t
conda env update --file environment.yaml --prune
```

### If Update leads to cudaKernelExc Error
```
conda create --name b2t pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda activate b2t
conda install scipy pydantic tokenizers transformers matplotlib
conda install -c conda-forge wandb
``` 

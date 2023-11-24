from tokenizers.trainers import BpeTrainer
from tokenizers.models import BPE
import os
from tokenizers import Tokenizer
import json
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path


def getTokenizer(
    trainFile: str = "/hpi/fs00/scratch/leon.hermann/b2t/allSentences.txt",
    outputFolder: str = "/hpi/fs00/scratch/leon.hermann/b2t/tokenizer",
    retrain: bool = False,
    **train_args,
) -> Tokenizer:
    tokenizer_prefix = "b2t_tokenizer"
    if os.path.exists(outputFolder) and not retrain:
        vocabPath = Path(outputFolder) / f"{tokenizer_prefix}-vocab.json"
        mergesPath = Path(outputFolder) / f"{tokenizer_prefix}-merges.txt"

        tokenizer = Tokenizer(
            BPE.from_file(vocab=str(vocabPath), merges=str(mergesPath))
        )
        tokenizer.pre_tokenizer = Whitespace()

        print("Got tokenizer from file")
    else:
        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = Whitespace()

        print(f"Started training tokenizer with train args: {train_args}")
        trainer = BpeTrainer(
            special_tokens=["<pad>", "<s>", "</s>", "<unk>"],
            max_token_length=1,
            limit_alphabet=256,
        )

        tokenizer.train(files=[trainFile], trainer=trainer)

        print("Finished training tokenizer")

        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)

        output_files = tokenizer.model.save(outputFolder, f"{tokenizer_prefix}")

        print(f"Wrote tokenizer to {output_files}")

    return tokenizer

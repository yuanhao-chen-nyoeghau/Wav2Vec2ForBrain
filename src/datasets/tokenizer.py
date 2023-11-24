from tokenizers.trainers import BpeTrainer
from tokenizers.models import BPE
import os
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path


def get_tokenizer(
    train_file: str = "/hpi/fs00/scratch/leon.hermann/b2t/allSentences.txt",
    output_folder: str = "/hpi/fs00/scratch/leon.hermann/b2t/tokenizer",
    retrain: bool = False,
    **train_args,
) -> Tokenizer:
    """Loads tokenizer from file if they are present or trains tokenizer and
        writes the vocabulary and merges to file.

    Args:
        train_file (str, optional): Path to all sentences in a txt file.
            Can be generated with scripts/writeAllSentencesToFile.py.
            Defaults to "/hpi/fs00/scratch/leon.hermann/b2t/allSentences.txt".
        output_folder (str, optional): Folder to write vocabulary and merges file to.
            Defaults to "/hpi/fs00/scratch/leon.hermann/b2t/tokenizer".
        retrain (bool, optional): Forces retraining. Defaults to False.
        train_args: Keyword args that set the train options for tokenizer training.
            See https://huggingface.co/docs/tokenizers/api/trainers#tokenizers.trainers.BpeTrainer

    Returns:
        Tokenizer: Return trained tokenizer with BPE model
    """
    tokenizer_prefix = "b2t_tokenizer"
    if os.path.exists(output_folder) and not retrain:
        vocab_path = Path(output_folder) / f"{tokenizer_prefix}-vocab.json"
        merges_path = Path(output_folder) / f"{tokenizer_prefix}-merges.txt"

        tokenizer = Tokenizer(
            BPE.from_file(vocab=str(vocab_path), merges=str(merges_path))
        )
        tokenizer.pre_tokenizer = Whitespace()

        print("Got tokenizer from file")
    else:
        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = Whitespace()

        print(f"Started training tokenizer with train args: {train_args}")
        trainer = BpeTrainer(
            special_tokens=["<pad>", "<s>", "</s>", "<unk>"],
            **train_args,
        )

        tokenizer.train(files=[train_file], trainer=trainer)

        print("Finished training tokenizer")

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        output_files = tokenizer.model.save(output_folder, f"{tokenizer_prefix}")

        print(f"Wrote tokenizer to {output_files}")

    return tokenizer

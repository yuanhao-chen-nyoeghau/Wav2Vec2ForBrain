from tokenizers.trainers import BpeTrainer
from tokenizers.models import BPE
import os
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
import numpy as np
from scipy.io import loadmat


def write_all_sentences_to_file(
    data_dir: str,
    output_file: str,
    suffixes: list[str] = ["train", "test"],
):
    all_sentences = np.array([])

    for suffix in suffixes:
        subfolder_path = Path(data_dir) / suffix

        data_files = [
            loadmat(Path(subfolder_path) / file_name)
            for file_name in os.listdir(subfolder_path)
        ]

        for data_file in data_files:
            fileSentences = data_file["sentenceText"]

            all_sentences = np.concatenate([all_sentences, fileSentences])

    parent_folder = os.path.dirname(output_file)
    if not os.path.exists(parent_folder):
        os.makedirs(parent_folder)

    with open(output_file, "w") as f:
        for sentence in all_sentences:
            f.write(sentence)

    print(f"Wrote {len(all_sentences)} sentences to {output_file}")


def get_tokenizer(
    train_file: str,
    dataset_splits_dir: str,
    tokenizer_config_dir: str,
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
    if os.path.exists(tokenizer_config_dir) and not retrain:
        vocab_path = Path(tokenizer_config_dir) / f"{tokenizer_prefix}-vocab.json"
        merges_path = Path(tokenizer_config_dir) / f"{tokenizer_prefix}-merges.txt"

        tokenizer = Tokenizer(
            BPE.from_file(vocab=str(vocab_path), merges=str(merges_path))
        )
        tokenizer.pre_tokenizer = Whitespace()

        print("Got tokenizer from file")
    else:
        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = Whitespace()

        if not os.path.exists(train_file):
            # Writes all sentences to file if it does not exist
            write_all_sentences_to_file(
                data_dir=dataset_splits_dir, output_file=train_file
            )

        print(f"Started training tokenizer with train args: {train_args}")
        trainer = BpeTrainer(
            special_tokens=["<pad>", "<s>", "</s>", "<unk>"],
            **train_args,
        )

        tokenizer.train(files=[train_file], trainer=trainer)

        print("Finished training tokenizer")

        if not os.path.exists(tokenizer_config_dir):
            os.makedirs(tokenizer_config_dir)

        output_files = tokenizer.model.save(tokenizer_config_dir, f"{tokenizer_prefix}")

        print(f"Wrote tokenizer to {output_files}")

    return tokenizer

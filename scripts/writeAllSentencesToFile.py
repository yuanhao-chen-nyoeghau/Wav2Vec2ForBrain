import json
import os
from pathlib import Path
from scipy.io import loadmat
import numpy as np


def writeAllSentencesToFile(
    dataFolder: str = "/hpi/fs00/scratch/florian.mueller/data/competitionData/",
    suffixes: list[str] = ["train", "test", "competitionHoldOut"],
    outputPath: str = "/hpi/fs00/scratch/leon.hermann/b2t/allSentences.txt",
):
    allSentences = np.array([])

    for suffix in suffixes:
        subfolderPath = Path(dataFolder) / suffix

        dataFiles = [
            loadmat(Path(subfolderPath) / fileName)
            for fileName in os.listdir(subfolderPath)
        ]

        for dataFile in dataFiles:
            fileSentences = dataFile["sentenceText"]

            allSentences = np.concatenate([allSentences, fileSentences])

    parent_folder = os.path.dirname(outputPath)
    if not os.path.exists(parent_folder):
        os.makedirs(parent_folder)

    with open(outputPath, "w") as f:
        for sentence in allSentences:
            f.write(sentence)

    print(f"Wrote {len(allSentences)} to {outputPath}")


if __name__ == "__main__":
    writeAllSentencesToFile()

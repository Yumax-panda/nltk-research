from nltk.corpus import brown
from nltk import NgramTagger
import matplotlib.pyplot as plt
import pandas as pd

Tagged = list[list[tuple[str, str]]]


def run(n: int, train: Tagged, test: Tagged) -> float:
    """n-gram タガーを実行する

    Parameters
    ----------
    n : int
        n-gram の n
    train : Tagged
        学習用データ
    test : Tagged
        テスト用データ

    Returns
    -------
    float
        精度
    """
    tagger = NgramTagger(n, train)
    return tagger.accuracy(test)


if __name__ == "__main__":
    tagged_sents = brown.tagged_sents(categories="news")
    size = int(len(tagged_sents) * 0.9)
    train = tagged_sents[:size]
    test = tagged_sents[size:]
    ns = range(1, 6)
    accuracies = [run(n, train, test) for n in ns]
    df = pd.DataFrame({"n": ns, "accuracy": accuracies})
    plt.plot(ns, accuracies)
    plt.xlabel("n")
    plt.ylabel("accuracy")
    plt.savefig("./n_tagger.png")
    print(df)

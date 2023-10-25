from typing import TypeVar
from nltk.corpus import brown
from nltk import (
    SequentialBackoffTagger,
    DefaultTagger,
    UnigramTagger,
    BigramTagger,
    RegexpTagger,
    FreqDist,
)
import json

TaggerT = TypeVar("TaggerT", bound=SequentialBackoffTagger)
Tagged = list[list[tuple[str, str]]]


# 名前を変更しているだけ
class CombinedTagger(BigramTagger):
    pass


class UnigramTaggerWithAllData(UnigramTagger):
    pass


def display_info(category: str) -> dict[str, int]:
    """データの情報を表示する

    Parameters
    ----------
    category : str
        カテゴリ

    Returns
    -------
    dict[str, int]
        データの情報
    """
    brown_sents = brown.sents(categories=category)
    brown_tagged_sents = brown.tagged_sents(categories=category)
    size = int(len(brown_tagged_sents) * 0.9)
    train_sents = brown_tagged_sents[:size]
    test_sents = brown_tagged_sents[size:]
    print(f"Category: {category}")
    print(f"Total sentences: {len(brown_sents)}")
    print(f"train_sents: {len(train_sents)}")
    print(f"test_sents: {len(test_sents)}\n")
    return {
        "total_sentences": len(brown_sents),
        "train_sents": len(train_sents),
        "test_sents": len(test_sents),
    }


def execute(tagger: TaggerT, tokens: list[str], answer: Tagged) -> dict[str, float]:
    """共通処理
    与えられたタガーでタグ付けを行い、その結果を表示する

    Parameters
    ----------
    tagger : TaggerT
        タグ付けで使用するタガー
    tokens : list[str]
        タグ付けするトークン
    answer : Tagged
        正解のタグ付け

    Returns
    -------
    dict[str, float]
        タガーの精度
    """
    name = tagger.__class__.__name__
    tagged = tagger.tag(tokens)
    accuracy = tagger.accuracy(answer)
    print(f"-----Tagger: {name}-----")
    print(f"Processed: {tagged[:10]}...")
    print(f"Accuracy: {accuracy}\n\n")
    return {name: accuracy}


def get_most_freq_tag(tagged_words: list[str]) -> str:
    """最も頻出するタグを返す

    Parameters
    ----------
    tagged_words : list[str]
        タグ付き単語

    Returns
    -------
    str
        最も頻出するタグ
    """
    return FreqDist(tag for (_, tag) in tagged_words).max()


def run(category: str) -> dict[str, float]:
    print(f"-----Category: {category}-----")

    # 実験に使用するデータ
    brown_tagged_sents = brown.tagged_sents(categories=category)
    sents = brown.sents(categories=category)
    tokens = sents[3]
    tagged_words = brown.tagged_words(categories=category)

    freq_tag = get_most_freq_tag(tagged_words)
    print(f"Most frequent tag: {freq_tag}")

    # デフォルトタガー
    backoff = freq_tag
    default_tagger = DefaultTagger(backoff)

    # 正規表現タガー
    patterns = [
        (r".*ing$", "VBG"),  # 動名詞
        (r".*ed$", "VBD"),  # 単純過去
        (r".*es$", "VBZ"),  # 三人称現在
        (r".*ould$", "MD"),  # 法助動詞 would, should...
        (r".*\'s$", "NN$"),  # 名詞の所有格
        (r".*s$", "NNS"),  # 名詞の複数形
        (r"^-?[0-9]+(.[0-9]+)?$", "CD"),  # 基数
        (r".*", "NN"),  # デフォルト　名詞
    ]
    regex_tagger = RegexpTagger(patterns, backoff=default_tagger)

    # ユニグラムタグ付け
    size = int(len(brown_tagged_sents) * 0.9)
    train_sents = brown_tagged_sents[:size]
    test_sents = brown_tagged_sents[size:]
    unigram_tagger_with_all_data = UnigramTaggerWithAllData(brown_tagged_sents)
    unigram_tagger = UnigramTagger(train_sents)

    # バイグラムタグ付け
    bigram_tagger = BigramTagger(train_sents)

    # バックオフを利用
    t1 = UnigramTagger(train_sents, backoff=default_tagger)
    t2 = BigramTagger(train_sents, backoff=t1)
    combined_tagger = CombinedTagger(train_sents, backoff=t2)

    data = {}

    # 実験
    for tagger in (
        default_tagger,
        regex_tagger,
        unigram_tagger_with_all_data,
    ):
        data.update(execute(tagger, tokens, brown_tagged_sents))

    for tagger in (unigram_tagger, bigram_tagger, combined_tagger):
        data.update(execute(tagger, tokens, test_sents))

    return {category: data}


if __name__ == "__main__":
    data = {}
    for category in {
        "news",
        "adventure",
        "belles_lettres",
        "editorial",
        "fiction",
        "government",
        "hobbies",
    }:
        data.update(run(category))
    with open("./result_short.json", "w") as f:
        json.dump(data, f, indent=4)

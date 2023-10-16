from typing import TypeVar
from nltk.corpus import brown
from nltk import (
    SequentialBackoffTagger,
    DefaultTagger,
    UnigramTagger,
    BigramTagger,
    RegexpTagger,
    FreqDist,
    ConditionalFreqDist,
)

TaggerT = TypeVar("TaggerT", bound=SequentialBackoffTagger)
Tagged = list[list[tuple[str, str]]]


# 名前を変更しているだけ
class LookUpTagger(UnigramTagger):
    pass


def execute(tagger: TaggerT, tokens: list[str], answer: Tagged):
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
    """
    name = tagger.__class__.__name__
    tagged = tagger.tag(tokens)
    accuracy = tagger.accuracy(answer)
    print(f"-----Tagger: {name}-----")
    print(f"Processed: {tagged[:10]}...")
    print(f"Accuracy: {accuracy}\n\n")


def get_lookup_tagger(
    words: list[str],
    tagged_words: list[str],
    lookup: int = 100,
    backoff: str = "NN",
) -> LookUpTagger:
    """lookupタガーを作成する

    Parameters
    ----------
    words : list[str]
        タガーの学習に使用する単語
    tagged_words : list[str]
        タガーの学習に使用するタグ付き単語
    lookup : int, optional
        lookupタガーのlookup数, by default 100
    backoff : str, optional
        lookupタガーのバックオフタガー, by default "NN"

    Returns
    -------
    UnigramTagger
        作成したlookupタガー
    """
    fd = FreqDist(words)
    cfd = ConditionalFreqDist(tagged_words)
    most_freq_words = fd.most_common(lookup)
    likely_tags = dict((word, cfd[word].max()) for (word, _) in most_freq_words)
    return LookUpTagger(model=likely_tags, backoff=DefaultTagger(backoff))


def main():
    # 実験に使用するデータ
    category = "news"
    brown_tagged_sents = brown.tagged_sents(categories=category)
    sents = brown.sents(categories=category)
    tokens = sents[3]
    words = brown.words(categories=category)
    tagged_words = brown.tagged_words(categories=category)

    # デフォルトタガー
    backoff = "NN"
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

    # lookupタガー
    lookup = 100
    lookup_tagger = get_lookup_tagger(
        words, tagged_words, lookup=lookup, backoff=backoff
    )

    # ユニグラムタグ付け
    size = int(len(brown_tagged_sents) * 0.9)
    train_sents = brown_tagged_sents[:size]
    test_sents = brown_tagged_sents[size:]
    unigram_tagger = UnigramTagger(train_sents)

    # バイグラムタグ付け
    bigram_tagger = BigramTagger(train_sents)

    for tagger in (default_tagger, regex_tagger, lookup_tagger):
        execute(tagger, tokens, brown_tagged_sents)
    execute(unigram_tagger, tokens, test_sents)


if __name__ == "__main__":
    main()

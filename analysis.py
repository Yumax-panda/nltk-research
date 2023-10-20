import json
import pandas as pd


def main(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["mean"] = df.mean(axis=1)
    print(df)
    print("\n")


if __name__ == "__main__":
    for path in {"./result.json", "./result_short.json"}:
        main(path)

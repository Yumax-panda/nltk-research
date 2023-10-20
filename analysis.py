import json
import pandas as pd


def main():
    with open("./result.json", "r") as f:
        data = json.load(f)
    categories = list(data.keys())
    df = pd.DataFrame(data)
    df["mean"] = df.mean(axis=1)

    print(f"Categories: {categories}\n\n")
    print(df)


if __name__ == "__main__":
    main()

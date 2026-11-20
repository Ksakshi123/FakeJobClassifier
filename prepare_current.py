"""Split collected current postings into train-extra and held-out eval sets.

- eval_current.csv  : ALL verbatim scam postings (never trained on) + a
                      sample of current legit postings. Used by evaluate.py
                      to measure real-world performance honestly.
- train_extra.csv   : synthetic scam postings (documented scam patterns,
                      used for augmentation) + remaining legit postings.
"""

import pandas as pd

LEGIT_PATH = "Dataset/current/legit_current.csv"
SCAM_PATH = "Dataset/current/scam_curated.csv"
N_EVAL_LEGIT = 150
SEED = 42


def main():
    legit = pd.read_csv(LEGIT_PATH)
    scam = pd.read_csv(SCAM_PATH)

    legit = legit.sample(frac=1, random_state=SEED).reset_index(drop=True)
    eval_legit = legit.iloc[:N_EVAL_LEGIT]
    train_legit = legit.iloc[N_EVAL_LEGIT:]

    verbatim = scam[scam["source"].str.startswith("verbatim")]
    synthetic = scam[scam["source"].str.startswith("synthetic")]

    eval_df = pd.concat([verbatim, eval_legit], ignore_index=True)
    train_df = pd.concat([synthetic, train_legit], ignore_index=True)

    eval_df.to_csv("Dataset/current/eval_current.csv", index=False)
    train_df.to_csv("Dataset/current/train_extra.csv", index=False)

    print(f"eval_current.csv : {len(eval_df)} rows "
          f"({int(eval_df['fraudulent'].sum())} scam, {int((eval_df['fraudulent']==0).sum())} legit)")
    print(f"train_extra.csv  : {len(train_df)} rows "
          f"({int(train_df['fraudulent'].sum())} scam, {int((train_df['fraudulent']==0).sum())} legit)")


if __name__ == "__main__":
    main()

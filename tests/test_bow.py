from datasets import load_dataset
import pandas as pd


def load_imdb_dataset():
    # We want a even disbtibution of positive and negative reviews
    dataset = load_dataset('imdb')
    df = pd.DataFrame(dataset['train'])

    pos_reviews = df[df['label'] == 1]
    neg_reviews = df[df['label'] == 0]

    N = 100
    pos_samples = pos_reviews.sample(N)
    neg_samples = neg_reviews.sample(N)

    balanced_dataset = pd.concat([pos_samples, neg_samples])
    balanced_dataset = balanced_dataset.sample(frac=1).reset_index(drop=True)
    dataset_dict = {col: balanced_dataset[col].tolist() for col in balanced_dataset.columns}
    balanced_dataset = Dataset.from_dict(dataset_dict)
    return balanced_dataset

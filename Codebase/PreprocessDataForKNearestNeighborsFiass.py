from datasets import load_dataset, Dataset, DatasetDict


def process():
    # Load the Datasets
    dataset = load_dataset("csv", data_files={"train": "../Datasets/processed_features_train.csv",
                                              "validation": "../Datasets/processed_features_dev.csv",
                                              "test": "../Datasets/processed_features_test.csv"})

    # Convert to Pandas
    dfs = {split: dset.to_pandas() for split, dset in dataset.items()}

    # Delete rows which the type of their review features are not string
    dfs["train"] = dfs["train"][dfs["train"]["review"].map(type) == str]
    dfs["validation"] = dfs["validation"][dfs["validation"]["review"].map(type) == str]
    dfs["test"] = dfs["test"][dfs["test"]["review"].map(type) == str]

    # Delete rows which their num_reviews are equal to 0
    dfs["train"] = dfs["train"][dfs["train"]["num_reviews"] != 0]
    dfs["test"] = dfs["test"][dfs["test"]["num_reviews"] != 0]
    dfs["validation"] = dfs["validation"][dfs["validation"]["num_reviews"] != 0]

    # Create the text column
    dfs["train"]["text"] = "<application package name>: " + dfs["train"]["app_name"].astype(str) \
                           + " <application review>: " + dfs["train"]["review"].astype(str) \
                           + " <number of words in the review>: " + dfs["train"]["Words Per Review"].astype(str) \
                           + " <application category>: " + dfs["train"]["category"].astype(str) \
                           + " <number of reviews for application>: " + dfs["train"]["num_reviews"].astype(str) \
                           + " <application price>: " + dfs["train"]["price"].astype(str) \
                           + " <application age rating>: " + dfs["train"]["rating"].astype(str)

    dfs["validation"]["text"] = "<application package name>: " + dfs["validation"]["app_name"].astype(str) \
                                + " <application review>: " + dfs["validation"]["review"].astype(str) \
                                + " <number of words in the review>: " + dfs["validation"]["Words Per Review"].astype(str) \
                                + " <application category>: " + dfs["validation"]["category"].astype(str) \
                                + " <number of reviews for application>: " + dfs["validation"]["num_reviews"].astype(str) \
                                + " <application price>: " + dfs["validation"]["price"].astype(str) \
                                + " <application age rating>: " + dfs["validation"]["rating"].astype(str)

    dfs["test"]["text"] = "<application package name>: " + dfs["test"]["app_name"].astype(str) \
                          + " <application review>: " + dfs["test"]["review"].astype(str) \
                          + " <number of words in the review>: " + dfs["test"]["Words Per Review"].astype(str) \
                          + " <application category>: " + dfs["test"]["category"].astype(str) \
                          + " <number of reviews for application>: " + dfs["test"]["num_reviews"].astype(str) \
                          + " <application price>: " + dfs["test"]["price"].astype(str) \
                          + " <application age rating>: " + dfs["test"]["rating"].astype(str)

    # Drop redundant columns except label
    dfs["train"].drop(
        ['app_name', 'review', 'votes', 'date', 'label_name', 'Words Per Review', 'category', 'num_reviews', 'price',
         'rating'], axis=1, inplace=True)
    dfs["validation"].drop(
        ['app_name', 'review', 'votes', 'date', 'label_name', 'Words Per Review', 'category', 'num_reviews', 'price',
         'rating'], axis=1, inplace=True)
    dfs["test"].drop(
        ['app_name', 'review', 'votes', 'date', 'label_name', 'Words Per Review', 'category', 'num_reviews', 'price',
         'rating'], axis=1, inplace=True)

    # Convert the Dataframe to a Dataset
    dataset = DatasetDict({split: Dataset.from_pandas(pdset, preserve_index=False) for split, pdset in dfs.items()})

    return dataset

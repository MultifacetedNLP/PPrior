from datasets import load_dataset


def process():
    # Load the Datasets
    dataset = load_dataset("csv", data_files={"train": "../Datasets/processed_features_train.csv",
                                              "validation": "../Datasets/processed_features_dev.csv",
                                              "test": "../Datasets/processed_features_test.csv"})

    # Convert to Pandas
    df_train = dataset["train"].to_pandas()
    df_val = dataset["validation"].to_pandas()

    # Delete rows which the type of their review features are not string
    df_train = df_train[df_train["review"].map(type) == str]
    df_val = df_val[df_val["review"].map(type) == str]

    # Delete rows which their num_reviews are equal to 0
    df_train = df_train[df_train["num_reviews"] != 0]
    df_val = df_val[df_val["num_reviews"] != 0]

    # Create the training_text column
    df_train["training_text"] = "<application package name>: " + df_train["app_name"].astype(str) \
                                + " <application review>: " + df_train["review"].astype(str) \
                                + " <number of words in the review>: " + df_train["Words Per Review"].astype(str) \
                                + " <application category>: " + df_train["category"].astype(str) \
                                + " <number of reviews for application>: " + df_train["num_reviews"].astype(str) \
                                + " <application price>: " + df_train["price"].astype(str) \
                                + " <application age rating>: " + df_train["rating"].astype(str)

    df_val["training_text"] = "<application package name>: " + df_val["app_name"].astype(str) \
                              + " <application review>: " + df_val["review"].astype(str) \
                              + " <number of words in the review>: " + df_val["Words Per Review"].astype(str) \
                              + " <application category>: " + df_val["category"].astype(str) \
                              + " <number of reviews for application>: " + df_val["num_reviews"].astype(str) \
                              + " <application price>: " + df_val["price"].astype(str) \
                              + " <application age rating>: " + df_val["rating"].astype(str)

    # Drop redundant columns except label
    df_train.drop(
        ['app_name', 'review', 'votes', 'date', 'label_name', 'Words Per Review', 'category', 'num_reviews', 'price',
         'rating'], axis=1, inplace=True)

    df_val.drop(
        ['app_name', 'review', 'votes', 'date', 'label_name', 'Words Per Review', 'category', 'num_reviews', 'price',
         'rating'], axis=1, inplace=True)

    return df_train, df_val

import random
from sentence_transformers import InputExample, losses
from torch.utils.data import DataLoader
import math


def get_positives(df_train, label, num_pos, con_lambda):
    df_train_current = df_train[df_train["label"] == label].sort_values(by="votes")
    current_vote = -1

    for _, row in df_train_current.iterrows():

        if current_vote != row["votes"]:
            df_train_select = df_train_current[((df_train_current["votes"] - row["votes"]) > -con_lambda) &
                                                ((df_train_current["votes"] - row["votes"]) < con_lambda)]
            current_vote = row["votes"]

        selected_rows_idx = random.sample(range(len(df_train_select)), num_pos)

        for selected_row in selected_rows_idx:
            yield InputExample(texts=[row['training_text'], df_train_select.iloc[selected_row]["training_text"]], label=1)


def get_negatives(df_train, label, num_neg_per_class, k, con_lambda):
    df_train_current = df_train[df_train["label"] == label].sort_values(by="votes")
    df_train_neg = df_train[df_train["label"] != label]
    current_vote = -1

    for _, row in df_train_current.iterrows():

        for _ in range(k):

            if current_vote != row["votes"]:
                df_train_select = df_train_neg[((df_train_neg["votes"] - row["votes"]) >= con_lambda) |
                                                ((df_train_neg["votes"] - row["votes"]) <= -con_lambda)]
                current_vote = row["votes"]

            selected_rows_idx = random.sample(range(len(df_train_select)), num_neg_per_class)

            for selected_row in selected_rows_idx:
                yield InputExample(texts=[row['training_text'], df_train_select.iloc[selected_row]["training_text"]], label=0)


def train(df_train, model, batch_size, epochs, output_path, num_classes, con_lambda=4):
    train_samples = []

    for label in range(num_classes):

        print("fetching positive samples...")

        for pos_sample in get_positives(df_train, label, 1, con_lambda):
            train_samples.append(pos_sample)

        print("fetching negative samples...")

        for neg_sample in get_negatives(df_train, label, 1, 4, con_lambda):
            train_samples.append(neg_sample)

    training_loader = DataLoader(train_samples, batch_size=batch_size, shuffle=True)
    training_loss = losses.ContrastiveLoss(model=model)
    warmup_steps = math.ceil(len(training_loader) * epochs * 0.1)

    # Train the sentence_transformers model
    model.fit(
        train_objectives=[(training_loader, training_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
        show_progress_bar=True
    )

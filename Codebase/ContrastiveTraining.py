import random
from sentence_transformers import InputExample, SentenceTransformer, models, losses
from torch.utils.data import DataLoader
import math


def get_positives(df_train, label, num_pos):
    df_train_current = df_train[df_train["label"] == label]

    for _, row in df_train_current.iterrows():

        selected_rows_idx = random.sample(range(len(df_train_current)), num_pos)

        for selected_row in selected_rows_idx:
            yield InputExample(texts=[row['training_text'], df_train_current.iloc[selected_row]["training_text"]],
                               label=1)


def get_negatives(df_train, label, num_neg_per_class):
    df_train_current = df_train[df_train["label"] == label]
    df_train_neg_list = []

    for neg_label in [0, 1, 2, 3, 4]:
        if neg_label != label:
            df_train_neg_list.append(df_train[df_train["label"] == neg_label])

    for _, row in df_train_current.iterrows():

        # for df_train_neg in df_train_neg_list:
        df_train_neg = df_train_neg_list[random.choice(range(4))]

        selected_rows_idx = random.sample(range(len(df_train_neg)), num_neg_per_class)

        for selected_row in selected_rows_idx:
            yield InputExample(texts=[row['training_text'], df_train_neg.iloc[selected_row]["training_text"]], label=0)


def train(df_train, model, batch_size, epochs, output_path):
    train_samples = []

    for label in [0, 1, 2, 3, 4]:

        print("fetching positive samples...")

        for pos_sample in get_positives(df_train, label, 1):
            train_samples.append(pos_sample)

        print("fetching negative samples...")

        for neg_sample in get_negatives(df_train, label, 1):
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

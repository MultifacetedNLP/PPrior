import PreprocessDataForSelfSupervision
import PreprocessDataForKNearestNeighborsFiass
import PreprocessDataForContrastiveTraining
import ContrastiveTraining
from KNearestNeighborsFaiss import FaissKNeighbors, train_knn, search_knn_first
from sentence_transformers import SentenceTransformer, models
import subprocess
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, precision_score, recall_score, \
    accuracy_score, cohen_kappa_score, matthews_corrcoef
import matplotlib.pyplot as plt
from Transformer import Transformer


# Create the Confusion Matrix of the classifier
def plot_confusion_matrix(y_preds, y_true, labels):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.show()


def show_metrics(y_preds, y_valid):
    f1 = f1_score(y_valid, y_preds, average="weighted")
    f1_macro = f1_score(y_valid, y_preds, average="macro")
    accuracy = accuracy_score(y_valid, y_preds)
    recall = recall_score(y_valid, y_preds, average="weighted")
    recall_macro = recall_score(y_valid, y_preds, average="macro")
    precision = precision_score(y_valid, y_preds, average="weighted")
    precision_macro = precision_score(y_valid, y_preds, average="macro")
    kappa = cohen_kappa_score(y_valid, y_preds)
    MCC = matthews_corrcoef(y_valid, y_preds)

    print(f"Weighted F1 Score: {f1}")
    print(f"Macro F1 Score: {f1_macro}")
    print(f"Accuracy Score: {accuracy}")
    print(f"Weighted Recall Score: {recall}")
    print(f"Macro Recall Score: {recall_macro}")
    print(f"Weighted Precision Score: {precision}")
    print(f"Macro Precision Score: {precision_macro}")
    print(f"Cohen's kappa Score: {kappa}")
    print(f"MCC Score: {MCC}")


def show_metrics_binary(y_preds, y_valid):
    f1 = f1_score(y_valid, y_preds)
    accuracy = accuracy_score(y_valid, y_preds)
    recall = recall_score(y_valid, y_preds)
    precision = precision_score(y_valid, y_preds)
    kappa = cohen_kappa_score(y_valid, y_preds)
    MCC = matthews_corrcoef(y_valid, y_preds)

    print(f"F1 Score of Class One: {f1}")
    print(f"Accuracy Score: {accuracy}")
    print(f"Recall Score of Class One: {recall}")
    print(f"Precision Score of Class One: {precision}")
    print(f"Cohen's kappa Score: {kappa}")
    print(f"MCC Score: {MCC}")


def RunSelfSupervision():
    PreprocessDataForSelfSupervision.process()
    subprocess.call(['sh', './SelfSupervision.sh'])


def RunContrastiveTraining():
    df_train, df_val = PreprocessDataForContrastiveTraining.process(binaryClassification=False)

    T5 = Transformer("../TrainedModels/PreTrainedT5", max_seq_length=512)
    pooler = models.Pooling(T5.get_word_embedding_dimension(), pooling_mode_mean_tokens=True)
    model = SentenceTransformer(modules=[T5, pooler])

    ContrastiveTraining.train(df_train, model, batch_size=8, epochs=1,
                              output_path="../TrainedModels/contrastive-training-pretrainedT5", num_classes=5)


def RunContrastiveTrainingBinary():
    df_train, df_val = PreprocessDataForContrastiveTraining.process(binaryClassification=True)

    T5 = Transformer("../TrainedModels/PreTrainedT5", max_seq_length=512)
    pooler = models.Pooling(T5.get_word_embedding_dimension(), pooling_mode_mean_tokens=True)
    model = SentenceTransformer(modules=[T5, pooler])

    ContrastiveTraining.train(df_train, model, batch_size=8, epochs=1,
                              output_path="../TrainedModels/contrastive-training-anomaly-pretrainedT5", num_classes=2)


def RunKNearestNeighbors(task):
    dataset, classes_size = PreprocessDataForKNearestNeighborsFiass.process(binaryClassification=False)
    batch_size = 10000
    kneighbor = FaissKNeighbors(768, path="../TrainedModels/FinalKNN", batch_size=batch_size, gpu=False,
                                classes_size=classes_size, num_classes=5)
    model = SentenceTransformer("../TrainedModels/contrastive-training-pretrainedT5", device='cuda')

    if task == "train":
        train_knn(kneighbor, model, dataset["train"], batch_size=batch_size)
    elif task == "search":
        result = search_knn_first(kneighbor, model, dataset["test"], K=101, nprobe=2000, batch_size=16)

        result.set_format("numpy")
        y_preds = result["predicted_label"]
        y_valid = result["label"]

        plot_confusion_matrix(y_preds, y_valid, ["not_serious", "getting_traction", "pay_attention", "serious", "hot"])
        show_metrics(y_preds, y_valid)


def RunKNearestNeighborsBinary(task):
    dataset, classes_size = PreprocessDataForKNearestNeighborsFiass.process(binaryClassification=True)
    batch_size = 10000
    kneighbor = FaissKNeighbors(768, path="../TrainedModels/FinalAnomalyKNN", batch_size=batch_size, gpu=False,
                                classes_size=classes_size, num_classes=2)
    model = SentenceTransformer("../TrainedModels/contrastive-training-anomaly-pretrainedT5", device='cuda')

    if task == "train":
        train_knn(kneighbor, model, dataset["train"], batch_size=batch_size)
    elif task == "search":
        result = search_knn_first(kneighbor, model, dataset["test"], K=101, nprobe=2000, batch_size=16)

        result.set_format("numpy")
        y_preds = result["predicted_label"]
        y_valid = result["label"]

        plot_confusion_matrix(y_preds, y_valid, ["not hot", "hot"])
        show_metrics_binary(y_preds, y_valid)


if __name__ == '__main__':
    task = input("Which task do you want to run? (A or B or C or D) \n A) Self Supervision \n B) Contrastive Training "
                 "\n C) Train K Nearest Neighbors \n D) Search K Nearest Neighbors \n")
    task = task.lower()

    if task == "a":
        RunSelfSupervision()
    elif task == "b":
        RunContrastiveTraining()  # RunContrastiveTrainingBinary()
    elif task == "c":
        RunKNearestNeighbors("train")  # RunKNearestNeighborsBinary("train")
    elif task == "d":
        RunKNearestNeighbors("search")  # RunKNearestNeighborsBinary("search")
    else:
        print("Incorrect input.")

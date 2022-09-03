# PPrior: Proactive Prioritization of App issues

Proactive Prioritization is a novel framework that can automatically predict the number of votes a particular mobile application review can receive.


In this GitHub repository, we provided the Python implementation of this framework. More specifically, this repository consists of the Codebase, Datasets, and Trained Models. The source code of this framework is in the Codebase folder.
The TrainedModels folder contains DVC files pointing to the actual models in Google Drive. Also, the Datasets.dvc file points to the real datasets folder in Google Drive.

# Datasets
Datasets can be downloaded from these links: [Train Dataset](https://drive.google.com/uc?export=download&id=1sfILpj4aAAjQ5eDmi4NSx3staDOqHQ1j), [Validation Dataset](https://drive.google.com/uc?export=download&id=1JbtZlQxXCaPf8dxMgWfxAIRIZttMcxcF), [Test Dataset](https://drive.google.com/uc?export=download&id=1YNE6a-uHTrkzmMy9bydZ5Wawt6BIFI6k)

# Setup
The steps for setting up the project on a local machine are as follows:

1- Clone the project

2- Install the [DVC](https://github.com/iterative/dvc) (Data Version Control) command line tool. The DVC is necessary for pulling the trained models and datasets from Google Drive to a local machine. 
The installation section can be found [here](https://github.com/iterative/dvc#installation).

3- Run the following command lines to pull trained models and datasets separately.

Pull Datasets:
```
dvc pull Datasets
```
Pull the PreTrainedT5 model:
```
dvc pull TrainedModels/PreTrainedT5
```
Pull the Multiclass SentenceTransformer model:
```
dvc pull TrainedModels/contrastive-training-pretrainedT5
```
Pull the Multiclass KNN index:
```
dvc pull TrainedModels/FinalKNN
```
Pull the Binary SentenceTransformer model:
```
dvc pull TrainedModels/contrastive-training-anomaly-pretrainedT5
```
Pull the Binary KNN index:
```
dvc pull TrainedModels/FinalAnomalyKNN
```
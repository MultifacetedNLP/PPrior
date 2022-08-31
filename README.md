# PPrior: Proactive Prioritization of App issues

Proactive Prioritization is a novel framework that can automatically predict the number of votes a particular mobile application review can receive.


In this GitHub repository, we provided the Python implementation of this framework. More specifically, this repository consists of source codes, datasets, and trained models.

# Setup
The steps for setting up the project on your local machine are as follows:

1- Clone the project

2- Install the [DVC](https://github.com/iterative/dvc) (Data Version Control) command line tool. The DVC is necessary for pulling the trained models and datasets from Google Drive to your local machine. You can find the installation section [here](https://github.com/iterative/dvc#installation).

3- Run the following command line to pull all the trained models and the datasets to your local machine.
```
dvc pull
```
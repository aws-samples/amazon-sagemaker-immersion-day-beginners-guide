# How to move your on-prem machine learning models into Amazon SageMaker - A Beginners Guide
This is the source code for [How to move your on-prem machine learning models into Amazon SageMaker - A Beginners Guide](https://apg-library.amazonaws.com/content-viewer/author/9e7e8490-c0e7-4273-b86f-22f1b0bb0b91).  
The guide demonstrates one way of approaching an Amazon SageMaker Immersion Day. It shows a step by step approach to migrate you machine learning (ML) models from any place into [Amazon SageMaker](https://aws.amazon.com/sagemaker/) to build, train and deploy you model. Further, the guide will discuss pipelining your ML solution with both approaches [AWS Step Functions](https://aws.amazon.com/step-functions/?step-functions.sort-by=item.additionalFields.postDateTime&step-functions.sort-order=desc) and [Amazon SageMaker Pipelines](https://aws.amazon.com/sagemaker/pipelines/). After reading this guide you can decide whether to run this approach during your Immersion Day. It can also be used in any other customer facing workshop or demo.


## File structure

```
.
├── ImmersionDay_Notebook1_GettingStarted.ipynb
├── ImmersionDay_Notebook2_SageMaker_Resources.ipynb
├── ImmersionDay_Notebook3_Pipelines.ipynb
├── ImmersionDay_Notebook4_StepFunctions.ipynb
├── README.md
└── src
    ├── preprocess.py
    └── train.py
```

## Contents
Repository is composed of 4 Jupyter Notebooks, each training model and inferring on it for Predictive Maintenance case.

- **Notebook 1** shows how to do Machine Learning inside Jupyter Notebook
- **Notebook 2** shows how to run the same code as in **Notebook 1** but this time using **SageMaker instances** and how to deploy **inference endpoint**
- **Notebook 3** adds orchestration with [**SageMaker Pipelines**](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-sdk.html) to **Notebook 2**
- **Notebook 4** adds orchestration with [**Step Functions**](https://docs.aws.amazon.com/step-functions/) to **Notebook 2**

## How to run the notebooks?
To run the notebooks clone them to your **SageMaker Notebook Instance**

## Authors
Selena Tabbara  
Michael Wallner  
Sanjay Ashok  
Mateusz Zaremba

## Helpful blogs to understand how to move from local Jupyter environment to Amazon SageMaker
- [Move from local jupyter to Amazon SageMaker — Part 1](https://medium.com/@pandey.vikesh/move-from-local-jupyter-to-amazon-sagemaker-part-1-7ef14af0fe9d)
- [Move from local jupyter to Amazon SageMaker — Part 2](https://medium.com/@pandey.vikesh/move-from-local-jupyter-to-amazon-sagemaker-part-2-f827832d4b9d)

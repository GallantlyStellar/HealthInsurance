# Raw Data

This is the source data that will be used for this project. The dataset is [publicly available](https://www.kaggle.com/datasets/willianoliveiragibin/healthcare-insurance/data) on Kaggle and is licensed to the public domain.

The dataset can be obtained from the Kaggle API with `curl --location --output ./insurance.zip https://www.kaggle.com/api/v1/datasets/download/willianoliveiragibin/healthcare-insurance`.

## DVC Note

These data are tracked with DVC. The underlying dataset is also committed into git for this project due to its relatively small size and to afford this repository with some independence so that viewers need not separately download the dataset when evaluating this project.

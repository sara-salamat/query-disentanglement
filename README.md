# Neural Disentanglement of Query Difficulty and Semantics

This repository contains the implementation and experiments for the paper titled "Neural Disentanglement of Query Difficulty and Semantics." The paper investigates the impact of factors beyond query semantics on retrieval effectiveness and presents a novel neural disentanglement method to separate query semantics from query difficulty. The disentangled representations enable determining semantic associations between queries and estimating query effectiveness.

## Methodology

Our methodology involves encoding queries into a latent space representation, separating them into distinct components for query difficulty and semantics. We leverage these components to determine semantic similarity between queries and compare their performance effectively. The following figure illustrate the overall process:
<p align="center">
  <img src="Framework.png" alt="diagram" width="500"/>
</p>

## Dataset
Our dataset comprises nearly 1 million pairs of query pairs, consisting of both similar and dissimilar queries. Each query pair is labeled to indicate whether the queries share similar semantics or not. The dataset is balanced, with an equal distribution of 50% for each label category.

You can download the dataset from the following link: [**Dataset Download**](https://drive.google.com/file/d/1f__GZLDefnv3BwP4WscLptBZgOz4NgAy/view?usp=sharing)

The dataset provides a valuable resource for training and evaluating models on tasks such as query performance prediction and semantic similarity calculation. Its balanced nature ensures a comprehensive coverage of both similar and dissimilar query scenarios, enabling robust analysis and accurate model assessment.

## Repository structure
* **`data/`**: Contains the datasets used in the experiments.
* **`output/`**: Trained models are saved in this directory
* **`Evaluate/`**: Evaluation scripts are in this folder. The code for calculating sMARE is also there.
* **`requirements.txt`**: List of Python dependencies required to run the code in this repository.

## Usage
1. Clone the repository and install the requirements.
2. Download the dataset and put it in `data/`.
3. For training a model, run `python TrainT.py`. This command trains a model with our default architecture and saves the model in `output/` directory.

To modify the model architecture, you can modify the `Model.py` script. To modify training process or change the encoder base, you can modify `TrainT.py` file.



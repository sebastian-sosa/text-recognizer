# Text Recognition App

The following project is based on the optical character recognition system of [Full Stack Deep Learning Bootcamp](https://fullstackdeeplearning.com). The purpose of the course is to describe the infrastructure, tooling, deployment and project structure needed for a deep learning production system.

## Description of the architecture

The prediction system is composed of an ensemble of neural networks: a [Fully Convolutional Network](https://arxiv.org/abs/1411.4038)  detects lines of text in the submitted picture, and each of these crops of lines of text is then passed through a sliding window CNN + LSTM model which detects the characters on each line of image.

![System architecture](./img/architecture.png)

## Project structure

The following diagram details the separation of concerns across the different modules.

```
  |
  +-- _data: Store raw and processed data used to train the models.
  +-- app
  |   +-- api: Deployable containerized web server that provides a REST API for inference.
  |   |   +-- __init__.py
  |   |   +-- app.py
  |   |   +-- Dockerfile
  |   |   +-- tests
  |   |
  |   +-- notebooks: IPython notebooks for dataset exploration.
  |   |
  |   +-- tasks: Convenience bash scripts for common tasks such as model trainig and testing, 
  |   |          container building and running, and running tests.
  |   |
  |   +-- text_recognizer
  |   |   +-- datasets: Provides abstractions to access datasets and their metadata.
  |   |   +-- models: Provides abstractions to train models and use them for inference
  |   |   +-- networks: Actual neural networks used by models. By separating the network architecture
  |   |   |             from other concerns such as output interpretation, data augmentation techniques,
  |   |   |             model evaluation logic, etc., we facilitate experimentation on different network
  |   |   |             architectures by simply replacing the network used by the model.
  |   |   +-- tests
  |   |   +-- weights: Store trained model weights.
  |   |   +-- __init__.py
  |   |   +-- character_predictor.py: Provides an API for recognizing a character on a given fixed-size image.
  |   |   +-- line_predictor.py: Provides an API for bounding lines of text on an image.
  |   |   +-- paragraph_predictor.py: Provides an API for recognizing text on a given image.
  |   |   +-- util.py
  |   |
  |   +-- training: Provides convenience scripts for running experiments.
  |   |
  |   +-- wandb: Stores experiments metadata
  |
  +-- Pipfile
  +-- Pipfile.lock
```

## Deployment

To deploy the containerized API in Google Run:

0. If necessary, install [Google Cloud SDK](https://cloud.google.com/sdk/install).
1. Build the docker image:  
  `cd text-recognizer/app`  
  `docker build -t text_recognizer_api -f api/Dockerfile .`
2. Push the image to [Google Container Registry](https://cloud.google.com/container-registry/docs/pushing-and-pulling).
3. On [Google Cloud Platform](https://console.cloud.google.com), go to Cloud Run and create a new service.  Provide the url of the pushed image as the Container image URL.

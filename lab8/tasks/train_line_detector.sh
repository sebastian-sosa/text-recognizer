#!/bin/bash
python training/run_experiment.py --gpu=0 --save '{"dataset": "IamParagraphsDataset", "model": "LineDetectorModel", "network": "fcn", "train_args": {"batch_size": 16, "epochs": 32}}'

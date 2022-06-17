# MLBAC-AdversarialAttack

`datasets/` directory contains two datasets forked from https://github.com/dlbac/DlbacAlpha/tree/main/dataset/synthetic.

`resnet.py` and `dataloader.py` contain source code of *ResNet* model architecture and train/test data preperation.

`CustomLowProFool.py` contains customized version of *LowProFool* algorithm that supports two APIs for attack simulation.

`MLBACAdversarialAttackSimilation.ipynb` is the Colab notebook file that helps to train initial model, load the model for the adversarial attack simulation, generate accessibility constraint, and imports the APIs from CustomLowProFool to simulate MLBAC adversarial attack.
 

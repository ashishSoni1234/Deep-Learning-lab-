# Deep Learning Lab

## Lab Experiments

### *Lab 1: Experiment 1 - Fully Connected Neural Network on MNIST*
In this experiment, we implemented a fully connected neural network for classifying digits from the MNIST dataset using NumPy. The key aspects covered include:
- Data preprocessing
- Model training with forward and backward propagation
- Evaluation using accuracy and loss metrics
- Data augmentation (random rotation, horizontal flipping) to improve generalization

### *Lab 2: Experiment 2 - Neural Networks on Linearly and Non-Linearly Separable Datasets*
This experiment focuses on training neural networks on linearly separable and non-linearly separable datasets (e.g., Moon, Circle) using NumPy. Key insights include:
- Understanding the limitations of networks without hidden layers
- Improvements achieved by adding hidden layers
- Comparing different activation functions (ReLU, Sigmoid) for better performance

### *Lab 3: Experiment 3 - Convolutional Neural Networks (CNNs) for Image Classification*
In this experiment, we implemented CNNs for classifying images from the Cats vs. Dogs and CIFAR-10 datasets. The key objectives include:
- Experimenting with different CNN architectures (number of convolutional layers, filter sizes, pooling layers, dropout, and batch normalization)
- Evaluating the impact of different configurations:
  - *Activation Functions:* ReLU, Tanh, Leaky ReLU
  - *Weight Initialization:* Xavier, Kaiming, Random
  - *Optimizers:* SGD, Adam, RMSprop
- Training and evaluating models using accuracy and loss metrics
- Comparing the best CNN models with a fine-tuned ResNet-18 model

## Repository Structure

Deep-Learning-Lab/
│── Experiment1/
│   ├── Experiment_1_Digit_Classification_Using_MNIST_Dataset.py
│   ├── Experiment_1_Digit_Classification_Using_MNIST_Dataset.ipynb
│   ├── bestWeights.npy 
│   ├── Documentation 
│── Experiment2/
│   ├── Experiment_2_Classify_Linearly_Separable_Dataset.py
│   ├── Experiment_2_Classify_Linearly_Separable_Dataset.ipynb 
│   ├── Documentation
│── Experiment3/
│   ├── Experiment_3_Classification_Using_Cat_Vs_Dog_Dataset/
│   │   ├── Experiment_3_Classification_Using_Cat_Vs_Dog_Dataset.py
│   │   ├── Experiment_3_Classification_Using_Cat_Vs_Dog_Dataset.ipynb
│   │   ├── best_model.pth
│   │   ├── Visualization_For_Cat_Vs_Dog.ipynb
│   │   ├── Accuracy_For_Cat_Vs_Dog.ipynb 
│   │   ├── Logs
│   ├── Experiment_3_Classification_Using_CIFAR_10_Dataset/
│   │   ├── Experiment_3_Classification_Using_CIFAR_10_Dataset.py
│   │   ├── Experiment_3_Classification_Using_CIFAR_10_Dataset.ipynb
│   │   ├── bestModel.pth 
│   │   ├── Visualization_For_CIFAR_10.ipynb 
│   │   ├── Logs
│   ├── Documentation
│── README.md


## Resources
- *MNIST Dataset*: [Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
- *Cats vs Dogs Dataset*: [Kaggle](https://www.kaggle.com/competitions/dogs-vs-cats)
- *CIFAR-10 Dataset*: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
- *Make Moons Dataset*: [Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html)
- *Make Circles Dataset*: [Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html)

# Machine Learning using Vision Transformer on CIFAR 10 Image Dataset

## Project Overview
This project implements and fine-tunes a Vision Transformer model for image classification on a subset of the CIFAR-10 dataset. The model was adapted from a pre-trained ViT-B/16 architecture and then retrained to classify images into 10 categories. This project also includes data preparation, model training, evaluation, and an end-to-end deployment to a Gradio web interface for real-time predictions.

## Project Structure

### Data Preparation
- Downloads CIFAR-10 dataset.
- Creates a custom subset with 100 training images and 25 test images per class.
- Implements data transformations and loading pipelines.

### Model Development
- Utilizes pre-trained ViT-B/16 model.
- Adapts the classification head for CIFAR-10 classes.
- Implements training and evaluation loops with PyTorch.

### Training and Evaluation
- Fine-tunes the model using `CrossEntropyLoss` and Adam optimizer.
- Tracks training metrics using TensorBoard.
- Evaluates model performance on the test set.
- Implements comprehensive prediction analysis.

### Deployment
- Creates an interactive web interface using Gradio.
- Enables real-time image classification.
- Provides prediction probabilities and inference time metrics.

## Key Features
- **Custom dataset creation from CIFAR-10**: Enables efficient and manageable experimentation.
- **Transfer learning with Vision Transformer**: Leverages a pre-trained ViT-B/16 model for faster convergence and better accuracy.
- **TensorBoard integration**: Tracks metrics such as loss, accuracy, and learning rates.
- **Interactive web interface**: Simplifies model testing and deployment with Gradio.
- **Comprehensive evaluation metrics**: Ensures detailed performance insights.
- **Efficient data processing pipeline**: Handles preprocessing, augmentation, and batching seamlessly.

## Acknowledgements
This project builds upon several key technologies and resources:
- CIFAR-10 dataset from the Canadian Institute For Advanced Research.
- Vision Transformer architecture from Google Research.
- PyTorch deep learning framework.
- Gradio library for creating ML web interfaces.
- Pre-trained ViT models from torchvision.
- One who helped me learn and implemet using pytorch. https://www.learnpytorch.io/, github: https://github.com/mrdbourke/pytorch-deep-learning/

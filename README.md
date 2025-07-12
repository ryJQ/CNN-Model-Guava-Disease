# CNN-Model-Guava-Disease
# Project Description
This project implements a Convolutional Neural Network (CNN) model using the ResNet50 architecture to classify diseases in guava fruits based on images. The model is trained on a labeled dataset of guava fruit images divided into training and testing sets. The trained model is then converted to TensorFlow Lite format for deployment on mobile applications.

# Key Features
* CNN model training with transfer learning using pretrained ResNet50 on ImageNet
* Data augmentation techniques to improve model generalization
* Handling class imbalance with class weights
* Model evaluation using accuracy, confusion matrix, and classification report
* Exporting the trained model to TensorFlow Lite (.tflite) for mobile deployment
* Visualization of correct and incorrect predictions on the test dataset

# How to Run
1. Prepare the dataset
* The dataset should have train and test directories, each containing subfolders for each guava disease class.
* Update the base_dir variable in the notebook to point to your dataset location.

2. Install dependencies
* Make sure you have Python 3.x installed along with the following packages:
```bash
pip install tensorflow matplotlib seaborn scikit-learn
```


3. Run the notebook
* Open CNN-Guava-Disease.ipynb in Jupyter Notebook or Visual Studio Code.
* Execute the cells sequentially to train the model, evaluate its performance, and export the model.

4. Use the TFLite model
* The .tflite model can be integrated into Android or other mobile platforms that support TensorFlow Lite for on-device inference.

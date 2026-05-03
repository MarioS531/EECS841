import os
import cv2 # sudo apt install python3-opencv for reading images
import numpy as np
import torch # sudo apt install python3-torch for deep learning features (system B)
import torchvision.models as models
from skimage.feature import hog # run sudo apt install python3-skimage to install scikit-image library for HoG feature extraction
from PIL import Image

# sudo apt install python3-sklearn
from sklearn.svm import SVC # import the SVM classifier
from sklearn.model_selection import train_test_split # to split the data into training and testing sets
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.model_selection import GridSearchCV

#load all images from dataset folder and assign labels based on filename
def load_dataset(dataset_path):
    images = []
    labels = []
    for file in sorted(os.listdir(dataset_path)):
        path = os.path.join(dataset_path, file)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)    #reads image as grayscale
        if img is None:
            continue
    
        if "happy" in file.lower():
            label = 0
        elif "angry" in file.lower():
            label = 1
        else:
            continue
        images.append(img)
        labels.append(label)
    return np.array(images), np.array(labels)

#System A feature extraction (HoG)
def extract_hog_features(images):
    features = []
    for img in images:
        hog_feature = hog(
            img,
            orientations=9,
            pixels_per_cell=(8,8),
            cells_per_block=(2,2),
            block_norm="L2-Hys" #normalizing method
        )
        features.append(hog_feature)
    return np.array(features)

def evaluation(actual, pred):
    # find accuracy, precision, and f1 scores for the actual vs predicted sets
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred)
    f1 = f1_score(actual, pred)
    return accuracy, precision, f1

def tune_params(X, y):
    # parameter tuning using grid search 
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto', 0.01, 0.001]
    }

    grid = GridSearchCV(
        SVC(),
        param_grid,
        cv=5, # 5 fold cross validation
        scoring='accuracy', # tuning based on best accuracy
        n_jobs=-1 # use all available cores
    )

    grid.fit(X, y)

    print("Best parameters:", grid.best_params_)
    print("Best accuracy:", grid.best_score_)

    return grid.best_estimator_ #return best model

def SystemA_classifier(hog_features, labels):
    # train and evaluate SVM as system A
    # train (80%), test (20%) split
    X_train, X_test, y_train, y_test = train_test_split(hog_features, labels, test_size=0.2)

    # define and train SVM model
    model = SVC()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)

    # evaluation metrics: accuracy, precision, f1-score for both training and testing sets (baseline)
    train_accuracy, train_prec, train_f1 = evaluation(y_train, y_train_pred)
    test_accuracy, test_prec, test_f1 = evaluation(y_test, y_pred)
    print("System A baseline evaluation: ")
    print("Training Accuracy:", train_accuracy)
    print("Testing Accuracy:", test_accuracy)
    print("Training Precision:", train_prec)
    print("Testing Precision:", test_prec)
    print("Training F1-Score:", train_f1)
    print("Testing F1-Score:", test_f1)

    # train new best model
    model = tune_params(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)

    # evaluation metrics: accuracy, precision, f1-score for both training and testing sets (tuned)
    train_accuracy, train_prec, train_f1 = evaluation(y_train, y_train_pred)
    test_accuracy, test_prec, test_f1 = evaluation(y_test, y_pred)
    print("System A tuned evaluation: ")
    print("Training Accuracy:", train_accuracy)
    print("Testing Accuracy:", test_accuracy)
    print("Training Precision:", train_prec)
    print("Testing Precision:", test_prec)
    print("Training F1-Score:", train_f1)
    print("Testing F1-Score:", test_f1)

def extract_ResNet_features(images):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Identity() # remove final classification layer to get features
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval() # set to evaluation mode

    preprocess = weights.transforms() # get preprocessing transforms for ResNet
    features = []
    print("Extracting ResNet features...")
    with torch.no_grad():
        for idx, img in enumerate(images):
            img_pil = Image.fromarray(img).convert("RGB") # convert to PIL image for preprocessing
            input_tensor = preprocess(img_pil).unsqueeze(0).to(device) # preprocess and add batch dimension
            feature = model(input_tensor).cpu().numpy().flatten() # extract features and flatten
            features.append(feature)
            if (idx+1) % 500 == 0:
                print(f"Processed {idx+1}/{len(images)} images")
    return np.array(features)

dataset_path = "Happy_Angry_Dataset_3K/dataset"
images, labels = load_dataset(dataset_path)
hog_features = extract_hog_features(images)
print("HoG feature shape:", hog_features.shape)

SystemA_classifier(hog_features, labels)

resnet_features = extract_ResNet_features(images) # placeholder for system B feature extraction
print("ResNet feature shape:", resnet_features.shape)
#SystemB_classifier(resnet_features, labels) # placeholder for system B classifier and evaluation

'''
System A results/discussion:
HoG feature shape: (3000, 900)
System A baseline evaluation: 
Training Accuracy: 0.9454166666666667
Testing Accuracy: 0.825
Training Precision: 0.9464579901153213
Testing Precision: 0.8082191780821918
Training F1-Score: 0.946068340881021
Testing F1-Score: 0.8180242634315424
Best parameters: {'C': 1, 'gamma': 'scale', 'kernel': 'rbf'}
Best accuracy: 0.7787499999999999 (based on CV)
System A tuned evaluation: 
Training Accuracy: 0.9454166666666667
Testing Accuracy: 0.825
Training Precision: 0.9464579901153213
Testing Precision: 0.8082191780821918
Training F1-Score: 0.946068340881021
Testing F1-Score: 0.8180242634315424

Tuning did not improve performance. Performance is already high, which may show that 
HoG does a good job at separating classes already so tuning the classifier is not needed.

The small gap between training and testing accuracy indicates mild overfitting.
'''

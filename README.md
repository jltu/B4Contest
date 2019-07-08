# Medical Image Segmentation Contest 2019

## The Task

Development of a bone metastasis detection algorithm in an anterior bone scintigram.

## Assessment Metrics

+ Evaluation Metrics: Dice similarity index (DSI) between the results for test cases and the bone metastasis labels.
+ DSI is calculated only inside the bone region.
+ Missing data will be scored as 0
+ Rank Computation Method
    + Mean DSI is used to order all participants.
    + If the two participants achieve the same score up to three significant digits, median of DSI will be used as a tie-breaker.

## Access the Data:
+ Dataset can be downloaded from \\\\tera\data\B4contest\2019.
+ The 68 patients of training data were released on May 26.
+ The 50 patients of test data will be released on June 24 (three days prior to the June 27 2019 deadline for submitting test predictions).

## Rules

+ Use one of the following classifier (Be sure to use a classifier different from others)
learn using the distributed bone scintigram and bone metastasis label.
+ Do not perform post-process on the results of recognition (the results depends only on the classifier's output).
+ When two or more participants use Deep Learning, make sure not to use the same network architecture.

    1.  k-NN classifier (Optimize “k”. It is required to scale features appropriately in advance or to employ Mahalanobis distance etc.)
    2.  Classification by likelihood of Gaussian mixture model approximation (Optimize the number of Gaussians. Use at least two Gaussians for each class.)
    3.  Neural network (Optimize the number of units/layers of neural network. When using DCNN etc., calculation of the features is not mandatory).
    4.  Support vector machine (Use kernel SVM and compare various kernel functions. Parameter optimization is required.)
    5.  Ensemble classifier (Compare various loss functions. This classifier also include Random Forest and Deep Forest)
    6.  Graph cuts (Calculation of the seed is important)
    7.  Classifiers other than the above or combinations of them classifiers


## Dataset

     dataset/
      ├─ case_list.txt  # List of the case names
      ├─ Tumor/         # Tumor labels
      ├─ Image/         # Bone scintigrams
      ├─ Bone/          # Bone structure labels
      ├─ Image_P/       # Bone scintigrams (posterior)
      └─ Bone_P/        # Bone structure labels (posterior)

## Result Submission Guidelines

     {your-name}.zip
     ├─ test_01.raw
     ├─ test_02.raw
     ⋮  
     └─ test_50.raw

+ Predictions should be submitted as a zip (your-name.zip) archive with the follwing format.
+ Pixel format must be 8bit binary image with pixel value 0 for background and 1 for tumor.

## Presentation:
+ Make a short presentation at 1:00 pm on June 28.
+ Include the following matters in the presentation:
    + Method
        + The entire flow and the detail of the algorithm
        + Description of the classifier
        + Debug
    + Results (training)
    + Discussions.


## Timeline
+ May  26: Training data release
+ June 24: Test data release
+ June 27: Result submission deadline
+ June 28: Presentation (from 1:00 pm) + after-party (from around 18:00 pm)


## Sample Codes
+ We provided the basic example of python scripts for segmentation and evaluation.
+ SimpleITK is required to run these examples.


## Organizers
+ Akinobu Shimizu
+ Atsushi Saito
+ Hayato Wakabayashi


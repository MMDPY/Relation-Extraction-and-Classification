# Intro to NLP - Assignment 5

## Team
|Student name| CCID |
|------------|------|
|student 1   |karimiab|
|student 2   |azamani1|

Please note that CCID is **different** from your student number.

## TODOs

In this file you **must**:
- [x] Fill out the team table above. 
- [x] Make sure you submitted the URL on eClass.
- [x] Acknowledge all resources consulted (discussions, texts, urls, etc.) while working on an assignment.
- [x] Provide clear installation and execution instructions that TAs must follow to execute your code.
- [x] List where and why you used 3rd-party libraries.
- [x] Delete the line that doesn't apply to you in the Acknowledgement section.

## Acknowledgement 
In accordance with the UofA Code of Student Behaviour, we acknowledge that  
(**delete the line that doesn't apply to you**)

- We did not consult any external resource for this assignment.
- We have listed all external resources we consulted for this assignment.

 Non-detailed oral discussion with others is permitted as long as any such discussion is summarized and acknowledged by all parties.

## 3-rd Party Libraries
You do not need to list `nltk` and `pandas` here.
* `main.py L:[4]` used `[sklearn.model_selection]` for [importing KFold class].
* `main.py L:[5]` used `[sklearn.metrics]` for [importing confusion_matrix class].
* `main.py L:[34]` used `[KFold]` for [defining the kf object for cross validation purposes].
* `main.py L:[232, 235, 236]` used `[confusion_matrix]` for [constructing the confusion matrix].

## Execution
Example usage: use the following command in the current directory.

`python3 src/main.py --train data/train.csv --test data/test.csv --output output/test.csv`

## Data

The assignment's training data can be found in [data/train.txt](data/train.txt),and the in-domain test data can be found in [data/test.txt](data/test.txt).

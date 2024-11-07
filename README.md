Insurance Claims Model
This project aims to predict insurance claims using machine learning. The model predicts two target variables:

Coverage Code
Accident Source
The project handles class imbalance using SMOTE (Synthetic Minority Over-sampling Technique) and applies class weights to ensure the model is robust against imbalanced data. The model uses embeddings of the claim descriptions as input features, generated by the SentenceTransformer.

The system includes a GUI to allow users to input their dataset and run the model, displaying evaluation results, and saving them to a file.

Project Structure
bash
Copy code
Insurance_Claims_Model/
├── input/
│   └── dataset_file.xlsx    # Example dataset file (replace with actual file)
├── output/
│   ├── embeddings/          # Store embeddings here
│   ├── model/               # Store trained models for Coverage Code and Accident Source here
│   └── reports/             # Store evaluation reports here
├── source_codes/
│   ├── model_training.py    # Code for model training
│   ├── gui.py               # Code for the GUI
│   └── utils.py             # Utility functions
└── README.md                # Project overview and instructions


Model Overview
Data:
Claim Description: The primary independent text feature.
Coverage Code: The first target variable (categorical).
Accident Source: The second target variable (categorical).


Preprocessing:
SMOTE is applied to address class imbalance.
Label Encoding is used to convert the categorical targets into numerical values.


Model:
RandomForestClassifier is used for both Coverage Code and Accident Source prediction.
The model is optimized using class weights and hyperparameter tuning via RandomizedSearchCV.


Evaluation:
The model's performance is evaluated using precision, recall, and ROC AUC scores for both target variables.
Results are stored in an Excel file.


Running the Model
1. Dataset Input
To run the model, you will need to provide a dataset containing the Claim Description, Coverage Code, and Accident Source columns. The dataset should be in .xlsx format.

2. Running the Model via GUI
Launch the GUI by running gui.py.
Click on Browse to select your dataset file (dataset_file.xlsx).
Click the Run button to train the model.
After the model runs, evaluation results will be displayed, and an Excel file containing these results will be saved in the output/reports folder.
3. Running the Model via Command Line (Optional)
You can also run the model directly from the command line using model_training.py. This will skip the GUI and run the model automatically.


python source_codes/model_training.py
The model will train on the data, evaluate the results, and save the model and evaluation results in the appropriate output directories.

Model Evaluation
After the model has run, the following evaluation metrics will be calculated for both targets:

Precision: The fraction of relevant instances among the retrieved instances.
Recall: The fraction of relevant instances that have been retrieved over the total relevant instances.
ROC AUC: The area under the receiver operating characteristic curve.
The evaluation results will be saved in an Excel file (evaluation_results.xlsx) in the output/reports folder.


# Predict Chemical Boiling Point
## Introduction 
This project goes through the ML project cycle of data collection, data pre-processing and feature enginneering, model training and validation. Five classic ML architectures, including Linear Regression, Random Forest, XGBoost, Neural Network, and Support Vector Regression, were trained and evaluated for the prediction of chemical boiling point.

## Data Collection
Chemical structure data is widely accessible online and I collected a large dataset of chemical compounds and their chemical properties from PubChem, a trusted and reputable chemical database. The file "compound_property_from_PubChem.csv" is downloaded from PubChem (https://pubchem.ncbi.nlm.nih.gov/). It contains 324,628 records. I mannually deleted some columns of the raw data file downloaded, mainly string type columns, to reduce the file size.

High-quality physical property data is less commonly available through public APIs. I sourced a smaller but well-documented dataset from a published academic journal article. The file "compound_boiling_points_from_literature.xlsx" is accessed from a literature (https://pubs.acs.org/doi/10.1021/acs.jchemed.3c01040). It contains 1748 records.

## Data Pre-processing and Exploratory Data Analysis
The two datasets collected were combined. With background knowledge, feature engineering was conducted to extract features that are potentially useful for predicting boiling points. EDA was conducted to get a good understanding of the dataset, like box plots and histograms of parameters.

The dataset contains outliers, specifically there is a small amount of data with high molecular weight and/or polar area and/or rotable bond number; those data are real though rare. In this project, I did not remove those outliers as we want the model to predict for those extreme cases too.

## Model Construction
* Approach
1. Split the data into training/validation/test.
2. Cross-validate/tune hyperparameters on the training dataset with GridSearchCV for each chosen ML architecyutre.
3. Compare the tuned model of each different architecture on the validation dataset and select the ‘champion’ ML architecture.
4. Retrain the champion model using training dataset + validation dataset, and further use that model to check the error on test dataset

* Five ML architectures are tested ,including
1. Linear regression
2. Random forest
3. XGBoost
4. Neural network
5. Support Vector Regressionk


## Summary results 

* Five classic ML models, including Linear Regression, Random Forest, XGBoost, Neural Network, and Support Vector Regression are evaluated for the problem of predicting chemical compounds boiling points on a total dataset of 1588 entries. Sensitivity of the model performance on data split was found thus it is recommaned to collect more data as a follow-up. In general, XGBoost model performed the best for this case. 
* Feature selection is naturally embedded with the training process of the XGBoost model. The model shows that major important features affecting boiling points for our dataset (all compouonds have no charge) are: mv (molecular weight), O_cnt(count of O atom), hbonddonor(Hydrogen Bond Donor Count), polararea, side_chain_cnt (count of side-chains), rotbonds(rotable bonds number).
* To further improve the model performance, collect more data (particularly more data with large polar area and/or rotbonds) to re-train the model.

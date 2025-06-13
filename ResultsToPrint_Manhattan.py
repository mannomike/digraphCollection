import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, precision_score, \
    recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import cityblock
from sklearn.metrics import confusion_matrix
import csv
from sklearn.multiclass import OneVsRestClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

'''
Mike Manno
Clarkson University
Practical Considerations for Keystroke Dynamics

To perform supervised Binary Classification
we use the 1 vs rest, where each user is a class, and we loop over each ID 
as Authenticated vs. the other classes, Not Authenticated.  

This is for Manhattan

'''

# File Name
df = pd.read_csv('1_AC_EA_HE.csv')

#X = df[['Dwell A', 'Dwell B', 'UD(AB)', 'DU(AB)', 'UU(AB)', 'DD(AB)']]  # Features
X = df[['Dwell A', 'Dwell B', 'UD(AB)', 'DU(AB)', 'UU(AB)', 'DD(AB)', 'Dwell A2', 'Dwell B2', 'UD(AB2)', 'DU(AB2)', 'UU(AB2)', 'DD(AB2)']]  # Features
#X = df[['Dwell A', 'Dwell B', 'UD(AB)', 'DU(AB)', 'UU(AB)', 'DD(AB)','Dwell A2', 'Dwell B2', 'UD(AB2)', 'DU(AB2)', 'UU(AB2)', 'DD(AB2)','Dwell A3', 'Dwell B3', 'UD(AB3)', 'DU(AB3)', 'UU(AB3)', 'DD(AB3)']]  # Features
y = df['ID']  # Target column

users = np.unique(y)  

def calculate_eer(fpr, tpr, thresholds):
    # EER occurs where FPR == 1 - TPR 
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer_threshold = thresholds[eer_idx]
    eer = fpr[eer_idx] 
    return eer, eer_threshold

for user in users:
    # Create binary target variable if it's the target user 1, or else 0
    y_binary = (y == user).astype(int)
    X_train, X_test, y_train, y_test = train_test_split( X, y_binary, test_size=0.3, random_state=42, stratify=y_binary)

    # Manhattan
    model = KNeighborsClassifier(n_neighbors=5, metric='manhattan')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    eer, eer_threshold = calculate_eer(fpr, tpr, thresholds)
    print(f"{user},{eer * 100:.2f}%,{roc_auc:.2f},{accuracy * 100:.2f}% ")

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    # Print the scores
    #print(f"Precision: {precision:.4f}")
    #print(f"Recall: {recall:.4f}")
    #print(f"F1-Score: {f1:.4f}")




print('Importing Libraries...')

# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier
import time
import joblib

sns.set_palette("colorblind")

chunk_size = 1000000
dtype_opt = {
    'frame.len': 'int64',
    'radiotap.dbm_antsignal': 'int64',
    'radiotap.length': 'int64',
    'radiotap.present.tsft': 'int64',
    'wlan.duration': 'int64',
    'wlan.fc.ds': 'int64',
    'wlan.fc.frag': 'int64',
    'wlan.fc.moredata': 'int64',
    'wlan.fc.protected': 'int64',
    'wlan.fc.pwrmgt': 'int64',
    'wlan.fc.type': 'int64',
    'wlan.fc.retry': 'int64',
    'wlan.fc.subtype': 'int64',
    'wlan_radio.duration': 'int64',
    'wlan_radio.signal_dbm': 'int64',
    'wlan_radio.phy': 'int64',
    'arp': 'object',
    'arp.hw.type': 'object',
    'arp.proto.type': 'int64',
    'arp.hw.size': 'int64',
    'arp.proto.size': 'int64',
    'arp.opcode': 'int64',
    'ip.ttl': 'int64',
    'tcp.analysis': 'int64',
    'tcp.analysis.retransmission': 'int64',
    'tcp.checksum.status': 'int64',
    'tcp.flags.syn': 'int64',
    'tcp.flags.ack': 'int64',
    'tcp.flags.fin': 'int64',
    'tcp.flags.push': 'int64',
    'tcp.flags.reset': 'int64',
    'tcp.option_len': 'int64',
    'udp.length': 'int64',
    'nbns': 'object',
    'nbss.length': 'int64',
    'ldap': 'object',
    'smb2.cmd': 'int64',
    'dns': 'object',
    'dns.count.answers': 'int64',
    'dns.count.queries': 'int64',
    'dns.resp.ttl': 'int64',
    'http.content_type': 'object',
    'http.request.method': 'object',
    'http.response.code': 'int64',
    'ssh.message_code': 'int64',
    'ssh.packet_length': 'int64'
}

# Read the data
print('Reading X...')
X = pd.DataFrame()
for chunk in pd.read_csv('/tf/notebooks/Notebooks/100%/X.csv', chunksize=chunk_size, usecols=dtype_opt.keys(), dtype=dtype_opt, low_memory=False):
    X = pd.concat([X, chunk])

print('Reading y...')
y = pd.DataFrame()
for chunk in pd.read_csv('/tf/notebooks/Notebooks/100%/y.csv', chunksize=chunk_size, usecols=['Label'], dtype='object', low_memory=False):
    y = pd.concat([y, chunk])

# Split the data into training and testing sets
print('Splitting the data...')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1234, stratify=y)
del X, y

# Scale the data
print('Scaling the data...')
scaler = MinMaxScaler()
scale_cols = ['frame.len',
        'radiotap.dbm_antsignal', 
        'radiotap.length', 
        'wlan.duration', 
        'wlan_radio.duration', 
        'wlan_radio.signal_dbm',
        'ip.ttl', 
        'udp.length', 
        'nbss.length',
        'dns.count.answers', 
        'dns.count.queries',
        'dns.resp.ttl',
        'ssh.packet_length']

X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test[scale_cols] = scaler.transform(X_test[scale_cols])

# Encode the labels
print('Encoding X...')
cols_to_encode = [col for col in X_train.columns if col not in scale_cols]
X_all = pd.concat([X_train, X_test], axis=0)
X_all_ohe = pd.get_dummies(X_all, columns=cols_to_encode, drop_first=True, dtype=np.uint8)
X_train_ohe = X_all_ohe[:len(X_train)]
X_test_ohe = X_all_ohe[len(X_train):]
del X_all
del X_all_ohe

print('Label Encoding y...')
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train.values.ravel())
y_test_encoded = le.transform(y_test.values.ravel())
del y_train, y_test


print('Training...')

start_time = time.time()
xgb = XGBClassifier(tree_method='gpu_hist', gpu_id=0, 
                    eval_metric='merror', early_stopping_rounds=10, 
                    subsample=0.9, n_estimators=200, min_child_weight=3, 
                    max_depth=9, learning_rate=0.3, gamma=0, colsample_bytree=0.7, verbose=1)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1234)

# Prepare arrays to store the metrics
aucs = []
f1s = []
precisions = []
recalls = []
accuracies = []

for train_index, test_index in skf.split(X_train_ohe, y_train_encoded):
    start_time = time.time()

    print('Training Fold....')

    X_train_fold, X_test_fold = X_train_ohe.iloc[train_index], X_train_ohe.iloc[test_index]
    y_train_fold, y_test_fold = y_train_encoded[train_index], y_train_encoded[test_index]
    
    xgb.fit(X_train_fold, y_train_fold, eval_set=[(X_test_fold, y_test_fold)], verbose=False)
    
    y_pred = xgb.predict(X_test_fold)
    y_pred_proba = xgb.predict_proba(X_test_fold)
    
    # Calculate and store the weighted metrics
    aucs.append(roc_auc_score(y_test_fold, y_pred_proba, multi_class='ovr', average='weighted'))
    f1s.append(f1_score(y_test_fold, y_pred, average='weighted'))
    precisions.append(precision_score(y_test_fold, y_pred, average='weighted'))
    recalls.append(recall_score(y_test_fold, y_pred, average='weighted'))
    accuracies.append(accuracy_score(y_test_fold, y_pred))
    
    print('Fold Metrics: ', "AUC: ", aucs[-1], "F1-score: ", f1s[-1], "Precision: ", precisions[-1], "Recall: ", recalls[-1], "Accuracy: ", accuracies[-1], "\n")
    elapsed_time = time.time() - start_time
    print(f'Time taken for fold: {elapsed_time:.2f} seconds')

# Calculate the average metrics
avg_auc = np.mean(aucs)
avg_f1 = np.mean(f1s)
avg_precision = np.mean(precisions)
avg_recall = np.mean(recalls)
avg_accuracy = np.mean(accuracies)

print("Average AUC:", avg_auc)
print("Average F1-score:", avg_f1)
print("Average Precision:", avg_precision)
print("Average Recall:", avg_recall)
print("Average Accuracy:", avg_accuracy)
elapsed_time = time.time() - start_time
print(f'Time taken for CV training: {elapsed_time:.2f} seconds')


# Train a final model on the entire training dataset and then evaluate its performance on the test set:
print('Training on entire training dataset...')
start_time = time.time()
eval_set = [(X_test_ohe, y_test_encoded)]
xgb.fit(X_train_ohe, y_train_encoded, eval_set=eval_set, verbose=True)
elapsed_time = time.time() - start_time
print(f'Time taken for final evaluation training: {elapsed_time:.2f} seconds')

print('Evaluating on test set...')

y_pred = xgb.predict(X_test_ohe)
predictions = [round(value) for value in y_pred]

# evaluate predictions

print('Test ROC AUC: ', roc_auc_score(y_test_encoded, xgb.predict_proba(X_test_ohe), multi_class='ovr'))
print('Test Precision: ', precision_score(y_test_encoded, y_pred, average='weighted'))
print('Test Recall: ', recall_score(y_test_encoded, y_pred, average='weighted'))
print('Test F1: ', f1_score(y_test_encoded, y_pred, average='weighted'))
print("Test Accuracy: ", accuracy_score(y_test_encoded, y_pred))

report = classification_report(y_test_encoded, y_pred)
print(report)

confusion = confusion_matrix(y_test_encoded, y_pred)
print('Confusion Matrix\n')
print(confusion)

# Plot the confusion matrix

labels = ['Botnet', 'Malware', 'Normal', 'SQL Injection', 'SDDP', 'SSH', 'Website Spoofing' ]
plt.figure(figsize=(10,10))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('cm.png')


filename = 'xgb.joblib'
joblib.dump(xgb, filename)

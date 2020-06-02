import json
import numpy as np
import pickle

def get(train_size, val_size, test_size):
    '''
    remember to adjust your file path
    read the data for use
    '''
    data_dir = '../../data/processed/'
    with open(data_dir+'label.json') as f:
        label = json.load(f)
    with open(data_dir+'non_sequential_features.json') as f:
        non_sequential_features = json.load(f)
#     with open(data_dir+'features_sequential_embedded.json') as f:
#         sequential_features = json.load(f)

    with open(data_dir+'features_sequential_embedded.txt', "rb") as fp:   #Unpickling
        feature1 = pickle.load(fp)

    label_size = len(label)
    train_size = int(train_size*label_size)
    val_size = int(val_size*label_size)
    test_size = int(test_size*label_size)
    
#     feature1 = np.array([sequential_features[key]
#                          for key in sequential_features.keys()])
    feature1 = np.array(feature1)
    feature2 = np.array([non_sequential_features[key]
                         for key in non_sequential_features.keys()])
    label = np.array([label[key] for key in label.keys()])
    feature1_benign = feature1[label == 1]
    feature2_benign = feature2[label == 1]
    label_benign = label[label == 1]

    idx = np.arange(len(label))
    np.random.shuffle(idx)

    idx_train = idx[:train_size]
    idx_val = idx[train_size:train_size+val_size]
    idx_test = idx[train_size+val_size:train_size+val_size+test_size]

    train_seq, val_seq, test_seq = feature1[idx_train], feature1[idx_val], feature1[idx_test]
    train_non, val_non, test_non = feature2[idx_train], feature2[idx_val], feature2[idx_test]
    train_Y, val_Y, test_Y = label[idx_train], label[idx_val], label[idx_test]

    return (train_seq, train_non, train_Y), (val_seq, val_non, val_Y), (test_seq, test_non, test_Y)

def get_KS(y_prob,y_true):
    ''' 
    y_prob: the predict_proba from the model
    y_true: the true label of Y
    this one calculates the KS value for a model
    return the best threshold, maximum ks
    '''
    from sklearn.metrics import roc_curve
    fpr,tpr,threshold=roc_curve(y_true,y_prob)
    ks=(tpr-fpr)
    max_=np.argmax(ks)
    
    return threshold[max_],np.max(ks)
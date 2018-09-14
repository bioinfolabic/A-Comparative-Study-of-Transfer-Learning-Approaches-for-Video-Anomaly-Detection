import numpy as np
from sklearn.svm import OneClassSVM 
import cv2
from sklearn.metrics import roc_curve, auc

def parameters_dict(x): # nu,gamma, maf mask size
    return {
        'avenue': [0.001,0.001,50],
        'avenue17': [0.01, 0.2, 5],
        'ped1': [0.1,0.01,150],
        'ped2': [0.01,1,100],
        'umn': [0.001, 0.01,150],
        'hsd1': [0.5,1,50],
        'hsd2': [0.01,0.01,50]
    }.get(x)

def test_video_sizes_dict(x):
    return {
        'avenue': [1439, 1211, 923, 947, 1007, 1283, 605, 36, 1175, 841, 472, 1272, 549, 506, 1001, 740, 426, 294, 248, 273, 76],
        'avenue17': [923, 947, 1007, 1283, 605, 472, 1272, 549, 506, 1001, 740, 426, 294, 248, 273, 76],
        'ped1': [200, 200, 200, 200, 200, 200, 200, 200, 200, 200],
        'ped2': [180,180,150,180,150,180,180,180,120,150,180,180],
        'umn': [130,148,459,645],
        'hsd1': [251, 224, 286, 290, 312, 297],
        'hsd2': [93, 241, 309, 180, 298, 565, 300]
    }.get(x)

def filter_scores(scores, dataset, mask):
	video_sizes = test_video_sizes_dict(dataset)
	new_test_scores = []
	init = 0
	for v in video_sizes:
		video = scores[init:init+v]
		video = cv2.blur(video, (1, mask))
		new_test_scores.extend(video)
		init = init+v
	return new_test_scores

def calc_auc(labels, scores):
	fpr, tpr, th = roc_curve(labels, scores, pos_label = 0)
	return auc(fpr,tpr)

def evaluate_features(train, test, labels, dataset, model=''):
        params = svm_parameters_dict(dataset)
        clf = OneClassSVM(kernel= 'rbf' , gamma = params[1], nu = params[0], verbose=True)
        clf.fit(train)
        decision_f = clf.decision_function(test)
        new_decision_f = filter_scores(decision_f, dataset, params[2])
        _auc = calc_auc(labels, new_decision_f)
        print "Area under ROC: ",_auc

def avenue_to_avenue17(test, labels):
        sizes = test_video_sizes_dict('avenue')
        labels_list = []
        for l in labels:
                labels_list.append(l)
        idx = 0
        videos = []
        for s in sizes:
                video = test[idx:idx+s]
        	video = np.squeeze(video)
                videos.append(video)
                idx = idx+s
        del_indexes = [0,1,7,8,9]
        for idx in sorted(del_indexes, reverse=True):
                del videos[idx]
                del labels_list[idx]
        labels = np.array(labels_list)
        test = np.vstack(videos)
        return test, labels

def load_features(dataset):
        if dataset == 'avenue17':
                path = 'features/avenue/'        
        else:
                path = 'features/'+dataset+'/'    
    
        train = np.load(path+'features_train.npy')
        test = np.load(path+'features_test.npy')
        labels = np.load(path+'labels.npy')

        if dataset == 'avenue17':
                test, labels = avenue_to_avenue17(test, labels)
        if dataset != 'hsd1' and dataset!='hsd2':
	        new_labels = []
	        for l in labels:
		        new_labels.extend(l)
	        labels = np.array(new_labels).astype(np.int)

        return train,test,labels

if __name__ == "__main__":
        import argparse
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('dataset', default='None', choices=['avenue', 'avenue17', 'ped1', 'ped2', 'umn', 'hsd1', 'hsd2'])
        args = parser.parse_args()
        dataset = args.dataset
        print args

        params = parameters_dict(dataset)

        train, test, labels = load_features(dataset)
        clf = OneClassSVM(kernel= 'rbf' , gamma = params[1], nu = params[0], verbose=True)
        clf.fit(train)
        decision_f = clf.decision_function(test)
        new_decision_f = filter_scores(decision_f, dataset, params[2])
        _auc = calc_auc(labels, new_decision_f)

        print "Area under ROC: ",_auc
















from scipy.stats import spearmanr
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import ppdb_utils
import numpy as np
import utils

def getSeqs(p1,p2,words):
    p1 = p1.split()
    p2 = p2.split()
    X1 = []
    X2 = []
    for i in p1:
        X1.append(ppdb_utils.lookupIDX(words,i))
    for i in p2:
        X2.append(ppdb_utils.lookupIDX(words,i))
    return X1, X2

def getSeqs2(p1,p2,words):
    p1 = p1.split()
    p2 = p2.split()
    p11 = [i.split('_')[0] for i in p1]
    p22 = [i.split('_')[0] for i in p2]
    sp11 = [i.split('_')[1] for i in p1]
    sp22 = [i.split('_')[1] for i in p2]
    X1 = []
    X2 = []
    SX1 = []
    SX2 = []
    for i in p11:
        X1.append(ppdb_utils.lookupIDX(words,i))
    for i in p22:
        X2.append(ppdb_utils.lookupIDX(words,i))
    for i in sp11:
	SX1.append(i)
    for i in sp22:
	SX2.append(i)
    return X1, X2, SX1, SX2

def getSeq(p1,words):
    p1 = p1.split()
    X1 = []
    for i in p1:
        X1.append(ppdb_utils.lookupIDX(words,i))
    return X1

def getCorrelation(model,words,f):
    f = open(f,'r')
    lines = f.readlines()
    preds = []
    golds = []
    seq1 = []
    seq2 = []
    for i in lines:
        i = i.split("\t")
        p1 = i[1]; p2 = i[2]; score = float(i[0])
	if len(p1.split()[0].split('_')) == 2:
	    X1, X2, SX1, SX2 = getSeqs2(p1,p2,words)
	else:
            X1, X2 = getSeqs(p1,p2,words)
        seq1.append(X1)
        seq2.append(X2)
        golds.append(score)
    x1,m1 = utils.prepare_data(seq1)
    x2,m2 = utils.prepare_data(seq2)
    scores = model.scoring_function(x1,x2,m1,m2)
    preds = np.squeeze(scores)
    return pearsonr(preds,golds)[0], spearmanr(preds,golds)[0]

def getCorrelation2(model,words,f):
    f = open(f,'r')
    lines = f.readlines()
    preds = []
    golds = []
    seq1 = []
    seq2 = []
    sseq1 = []
    sseq2 = []
    for i in lines:
        i = i.split("\t")
        p1 = i[1]; p2 = i[2]; score = float(i[0])
        X1, X2, SX1, SX2 = getSeqs2(p1,p2,words)
        seq1.append(X1)
        seq2.append(X2)
	sseq1.append(SX1)
	sseq2.append(SX2)
        golds.append(score)
    x1,m1,s1 = utils.prepare_data2(seq1,sseq1)
    x2,m2,s2 = utils.prepare_data2(seq2,sseq2)
    scores = model.scoring_function2(x1,x2,m1,m2,s1,s2)
    preds = np.squeeze(scores)
    return pearsonr(preds,golds)[0], spearmanr(preds,golds)[0]

def acc(preds,scores):
    golds = []
    for n,i in enumerate(scores):
        p = -1
        i=i.strip().lower()
        if i == "contradiction":
            p = 0
        elif i == "neutral":
            p = 1
        elif i == "entailment":
            p = 2
        else:
            raise ValueError('Something wrong with data...')
        golds.append(p)
    #print confusion_matrix(golds,preds)
    return accuracy_score(golds,preds)

def acc_para(preds,scores):
    golds = []
    for n,i in enumerate(scores):
        p = -1
        i=i.strip()
        if i == "0":
            p = 0
        elif i == "1":
            p = 1
        else:
            raise ValueError('Something wrong with data...')
        golds.append(p)
    #print confusion_matrix(golds,preds)
    return accuracy_score(golds,preds)

def accSentiment(preds,scores):
    golds = []
    for n,i in enumerate(scores):
        p = -1
        i=i.strip()
        if i == "0":
            p = 0
        elif i == "1":
            p = 1
        else:
            raise ValueError('Something wrong with data...')
        golds.append(p)
    return accuracy_score(golds,preds)

def getAcc(model,words,f):
    f = open(f,'r')
    lines = f.readlines()
    preds = []
    golds = []
    seq1 = []
    seq2 = []
    ct = 0
    for i in lines:
        i = i.split("\t")
        p1 = i[1]; p2 = i[2]; score = i[0]
	if len(p1.split()[0].split('_')) == 2:
	    X1, X2, SX1, SX2 = getSeqs2(p1,p2,words)
	else:
            X1, X2 = getSeqs(p1,p2,words)
        seq1.append(X1)
        seq2.append(X2)
        ct += 1
        if ct % 100 == 0:
            x1,m1 = utils.prepare_data(seq1)
            x2,m2 = utils.prepare_data(seq2)
            scores = model.scoring_function(x1,x2,m1,m2)
            scores = np.squeeze(scores)
            preds.extend(scores.tolist())
            seq1 = []
            seq2 = []
        golds.append(score)
    if len(seq1) > 0:
        x1,m1 = utils.prepare_data(seq1)
        x2,m2 = utils.prepare_data(seq2)
        scores = model.scoring_function(x1,x2,m1,m2)
        scores = np.squeeze(scores)
        preds.extend(scores.tolist())
    return acc(preds,golds)

def getAcc2(model,words,f):
    f = open(f,'r')
    lines = f.readlines()
    preds = []
    golds = []
    seq1 = []
    seq2 = []
    sseq1 = []
    sseq2 = []
    ct = 0
    for i in lines:
        i = i.split("\t")
        p1 = i[1]; p2 = i[2]; score = i[0]
        X1, X2, SX1, SX2 = getSeqs2(p1,p2,words)
        seq1.append(X1)
        seq2.append(X2)
	sseq1.append(SX1)
	sseq2.append(SX2)
        ct += 1
        if ct % 100 == 0:
            x1,m1,s1 = utils.prepare_data2(seq1, sseq1)
            x2,m2,s2 = utils.prepare_data2(seq2, sseq2)
            scores = model.scoring_function2(x1,x2,m1,m2,s1,s2)
            scores = np.squeeze(scores)
            preds.extend(scores.tolist())
            seq1 = []
            seq2 = []
	    sseq1 = []
	    sseq2 = []
        golds.append(score)
    if len(seq1) > 0:
        x1,m1,s1 = utils.prepare_data2(seq1, sseq1)
        x2,m2,s2 = utils.prepare_data2(seq2, sseq2)
        scores = model.scoring_function2(x1,x2,m1,m2,s1,s2)
        scores = np.squeeze(scores)
        preds.extend(scores.tolist())
    return acc(preds,golds)

def getAcc_para(model,words,f):
    f = open(f,'r')
    lines = f.readlines()
    preds = []
    golds = []
    seq1 = []
    seq2 = []
    ct = 0
    for i in lines:
        i = i.split("\t")
        p1 = i[1]; p2 = i[2]; score = i[0]
	if len(p1.split()[0].split('_')) == 2:
            X1, X2, SX1, SX2 = getSeqs2(p1,p2,words)
        else:
            X1, X2 = getSeqs(p1,p2,words)
        seq1.append(X1)
        seq2.append(X2)
        ct += 1
        if ct % 100 == 0:
            x1,m1 = utils.prepare_data(seq1)
            x2,m2 = utils.prepare_data(seq2)
            scores = model.scoring_function(x1,x2,m1,m2)
            scores = np.squeeze(scores)
            preds.extend(scores.tolist())
            seq1 = []
            seq2 = []
        golds.append(score)
    if len(seq1) > 0:
        x1,m1 = utils.prepare_data(seq1)
        x2,m2 = utils.prepare_data(seq2)
        scores = model.scoring_function(x1,x2,m1,m2)
        scores = np.squeeze(scores)
        preds.extend(scores.tolist())
    return acc_para(preds,golds)

def getAcc_para2(model,words,f):
    f = open(f,'r')
    lines = f.readlines()
    preds = []
    golds = []
    seq1 = []
    seq2 = []
    sseq1 = []
    sseq2 =[]
    ct = 0
    for i in lines:
        i = i.split("\t")
        p1 = i[1]; p2 = i[2]; score = i[0]
        X1, X2, SX1, SX2 = getSeqs2(p1,p2,words)
        seq1.append(X1)
        seq2.append(X2)
	sseq1.append(SX1)
	sseq2.append(SX2)
        ct += 1
        if ct % 100 == 0:
            x1,m1,s1 = utils.prepare_data2(seq1,sseq1)
            x2,m2,s2 = utils.prepare_data2(seq2,sseq2)
            scores = model.scoring_function2(x1,x2,m1,m2,s1,s2)
            scores = np.squeeze(scores)
            preds.extend(scores.tolist())
            seq1 = []
            seq2 = []
	    sseq1 = []
	    sseq2 = []
        golds.append(score)
    if len(seq1) > 0:
        x1,m1,s1 = utils.prepare_data2(seq1,sseq1)
        x2,m2,s2 = utils.prepare_data2(seq2,sseq2)
        scores = model.scoring_function2(x1,x2,m1,m2,s1,s2)
        scores = np.squeeze(scores)
        preds.extend(scores.tolist())
    return acc_para(preds,golds)

def getAccSentiment(model,words,f):
    f = open(f,'r')
    lines = f.readlines()
    preds = []
    golds = []
    seq1 = []
    ct = 0
    for i in lines:
        i = i.split("\t")
        p1 = i[0]; score = i[1]
        X1 = getSeq(p1,words)
        seq1.append(X1)
        ct += 1
        if ct % 100 == 0:
            x1,m1 = utils.prepare_data(seq1)
            scores = model.scoring_function(x1,m1)
            scores = np.squeeze(scores)
            preds.extend(scores.tolist())
            seq1 = []
        golds.append(score)
    if len(seq1) > 0:
        x1,m1 = utils.prepare_data(seq1)
        scores = model.scoring_function(x1,m1)
        scores = np.squeeze(scores)
        preds.extend(scores.tolist())
    return accSentiment(preds,golds)

def evaluate(model,words,file,params):
    if params.task == "sim":
        p,s = getCorrelation(model,words,file)
        return p,s
    elif params.task == "ent":
	s = getAcc(model,words,file)
	return s
    elif params.task == 'para':
        s = getAcc_para(model,words,file)
        return s
    elif params.task == "sentiment":
        s = getAccSentiment(model,words,file)
        return s
    else:
        raise ValueError('Task should be ent, sim, or sentiment')

def evaluate2(model,words,file,params):
    if params.task == "sim":
        p,s = getCorrelation2(model,words,file)
        return p,s
    elif params.task == "ent":
        s = getAcc2(model,words,file)
        return s
    elif params.task == 'para':
        s = getAcc_para2(model,words,file)
        return s
    elif params.task == "sentiment":
        s = getAccSentiment(model,words,file)
        return s
    else:
        raise ValueError('Task should be ent, sim, or sentiment')

def evaluate_all(model,words):
    prefix = "../data/"
    parr = []; sarr = []

    farr = ["annotated-ppdb-dev",
            "annotated-ppdb-test",
            "sicktest"]

    for i in farr:
        p,s = getCorrelation(model,words,prefix+i)
        parr.append(p); sarr.append(s)

    s = ""
    for i,j,k in zip(parr,sarr,farr):
        s += str(i)+" "+str(j)+" "+k+" | "

    print s

from django.shortcuts import render
from django.conf import settings
from django.http import HttpResponse
# Create your views here.
def index(request):
    return render(request,'index.html')

def predict(request):
	import pandas as pd
	import numpy as np 
	import re
	from sklearn.naive_bayes import MultinomialNB
	X_train = np.loadtxt('after_train.csv',dtype=int,delimiter=',')
	Y_train = np.loadtxt('y_train.csv',dtype=str,delimiter=',')
	X_test = np.zeros(1000)
	text = request.GET['textarea']
	words = re.compile("\w+").findall(text)
	feature_set = np.loadtxt('feature_set.csv',dtype=str,delimiter=',')
	feature_set=list(feature_set)
	stop_words=()
	vocab = dict()
	with open('stop_words.txt','r') as f:
	    word_stop=f.read().split()
	    stop_words=set(word_stop)
	X_test = np.zeros(1000)
	for word in words:
	    if word in feature_set:
	        X_test[feature_set.index(word)]+=1
	X_test = X_test.reshape((1,-1))
	ab=MultinomialNB(alpha=0.54)
	ab.fit(X_train,Y_train)
	y_pred_sk=ab.predict(X_test)
	response = y_pred_sk[0]
	return HttpResponse(response)


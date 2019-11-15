from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import cross_validation
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import operator

import numpy as np
import math





def euclideanDistance(instance1, instance2, length):
	distance = 0
	#print(instance1)
	#print(instance2)
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)





def getDistance(trainingSet,testInstance,k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	#print(distances)
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	#print(neighbors)
	#print(neighbors[2][-1])

	return neighbors


def prediction(neighbors):
	
	votes=[]
	for x in range(len(neighbors)):
		single_prediction=neighbors[x][-1]
		votes.append(single_prediction)
	
	freq=most_frequent(votes)
	return freq
	

def most_frequent(List): 
    counter = 0
    num = List[0] 
      
    for i in List: 
        curr_frequency = List.count(i) 
        if(curr_frequency> counter): 
            counter = curr_frequency 
            num = i 
  
    return num 


def accuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet)))*100





def class_probability_cal(result,number_of_labels):
	
	Prob_count=[0]*number_of_labels

	for y in range(0,len(result),1):
		for x in range(0,number_of_labels,1):
			if result[y]==x:
				Prob_count[x]=Prob_count[x]+1
				#print(count)
		
	Total_votes=0.0
	for w in range(0,number_of_labels,1):
		Total_votes=Total_votes+Prob_count[w]

	for w in range(0,number_of_labels,1):
		Prob_count[w]=Prob_count[w]/Total_votes
	return(Prob_count)
	
	
	
def sklearn_knn(train_data, test_data,k):
	clf= KNeighborsClassifier(n_neighbors=k) 
	#print(train_data[:, 4:5])
	clf.fit(train_data[:, 0:4],train_data[:,4:5])
	print('Accuracy of the Sklearn KNN classifier for k - '+str(k)+' is '+str(clf.score(test_data[:, 0:4],test_data[:,4:5])*100)+'%')



scaler = MinMaxScaler()
folds=10
kf = KFold(folds,True, 1)


iris =load_iris()

input_data=iris.data
target_data=iris.target
class_names = iris.target_names

train_data=[]
test_data=[]


train_labels=[]
test_labels=[]



result=[]

for x in range(0,len(input_data),1):
		result.append(np.append(input_data[x],target_data[x]))
data=result	

fold_rate=1
data=np.asarray(data)
for train,test in kf.split(data):
	train_data.append(data[train])
	test_data.append(data[test])

	#print(len(train_data))
	#print(len(test_data))

	upto_k_value=5								#K value
	number_of_labels=3


	for K in range(1,upto_k_value+1,1):

		sklearn_knn(data[train], data[test],K) #computing knn in sklearn

		pred=[]								#add new data to features_test in order to get the predictio
		for i in range(0,len(data[test])):

			neighbors=getDistance(data[train],data[test][i],K)#getting distance between one test point two all other trained points.
			result=prediction(neighbors)				#Getting k neighbouring points cordinates near testpoint.
			pred.append(result)
	

	

		Acu=accuracy(np.asarray(data[test]),np.asarray(pred))
		print('Accuracy of the my model for k - '+str(K)+' is '+str(Acu)+ '%')
		
		#print(confusion_matrix(data[test], pred))
	


		probability=class_probability_cal(pred,number_of_labels)
		print('probability of class1 class2 class3 in my model-'+str(probability))
		
		print('\n')

	print('KFold value -'+str(fold_rate))
	print('\n')
	fold_rate=fold_rate+1
	
	
#predi = np.asarray(pred)
#test=np.asarray(data[test][:,0:4])

#prediction=np.reshape(predi, (15, 1))
#data=np.reshape(test, (15, 1))
#print(test.resize(15,1))
#print(test)

print(data[test][:,4:5])
print(pred)
#print(np.asarray(data[test][:,0:4]))
cm=confusion_matrix(data[test][:,4:5],pred)
print(cm)
nor=scaler.fit(cm)
normalised=scaler.transform(cm)
print(normalised)






#install dependencies using "pip install ..." and save using "pip freeze > requirements.txt"
import os
import pickle

from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


#prepare the data
input_dir = ''; #dir with images
categories = ['empty', 'not_empty']

data = []
labels = []

for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, categories)):
        #read / save images
        img_path = os.path.join(input_dir, category, file)
        img = imread(img_path)
        #resize the image tp 15x15
        img = resize(img, (15, 15)) 
        #add to the data array
        img.append(img.flatten())
        labels.append(category_idx)

#cast the data into numpy arrays
data = np.asarray(data)
labels = np.asrray(data)

#train the model / split the data
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

#train classifier
classifier = SVC()

#train many classifiers depending on the "gamma" and "c" combinations
parameters = [{'gamma': [0.01, 0.001, 0.001], 'C': [1, 10, 100, 1000]}]

#trains the classifiers using the parameters passed
grid_search = GridSearchCV(classifier, parameters)

#pass the data to train the classifiers
grid_search.fit(X_train, y_train)

#test performance (get the best classifier created)
best_estimator = grid_search.best_estimator_

y_prediction = best_estimator.predict(X_test)
score = accuracy_score(y_prediction, y_test)

print('{}% of samples were successfully classified'.format(str(score * 100)))

#save the module
pickle.dump(best_estimator, open('./model.p', 'wb'))

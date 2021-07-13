from numpy.lib.histograms import histogram
from numpy.lib.type_check import imag
from scipy.sparse.construct import random
from utils import *
from sklearn.model_selection import train_test_split
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # This code is written to ignore the warnings


# STEP 1
path = 'SimulationData'
data = importDataInfo(path)
#print(data.head())


# STEP 2
balanceData(data, display=False)


# STEP 3
imagesPath, steerings = loadData(path, data)


# STEP 4
X_train, X_test, y_train, y_test = train_test_split(imagesPath, steerings, test_size=0.2, random_state=5)
print('Total Training Images: ', len(X_train))
print('Total Validation Images: ', len(X_test))


# STEP 5



# STEP 6



# STEP 7



# STEP 8
model = creatModel()
model.summary()


# STEP 9
history = model.fit(batchGen(X_train, y_train, 100, 1), steps_per_epoch=300, epochs=10, 
                            validation_data=batchGen(X_test, y_test, 100, 0), validation_steps=200)


# STEP 10
model.save('model.h5')
print('Model Saved')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
#plt.ylim([0,1])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()
"""
@Author: Marc Tunnell

Some or all of this code has been adapted from code provided by the following:
    https://github.com/vlawhern/arl-eegmodels/blob/master/examples/ERP.py
and:https://github.com/YundongWang/BCI_Challenge/blob/master/EEGNet.py
"""

import numpy as np
import pandas as pd
from EEGModels import EEGNet
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import roc_curve, auc
from EEGNet_Implementation.ERN_DS.data_preprocess import training, testing, total_stimulus, nodes, target_dimension


x = np.load('./BCI Challenge/Training.npy')
x = np.reshape(x, (training * total_stimulus, nodes, target_dimension))
y = pd.read_csv('./BCI Challenge/TrainLabels.csv')['Prediction'].values

x_test = np.load('./BCI Challenge/Testing.npy')

x_test = np.reshape(x_test, (testing * total_stimulus, nodes, target_dimension))

truth = np.reshape(pd.read_csv('./BCI Challenge/true_labels.csv', header=None).values, testing * total_stimulus)

x_train = x[1360:, :]
x_validate = x[:1360, :]
y_train = y[1360:]
y_valid = y[:1360]

kernels, chans, samples = 1, nodes, target_dimension

x_train = x_train.reshape(x_train.shape[0], chans, samples)
x_validate = x_validate.reshape(x_validate.shape[0], chans, samples)
x_test = x_test.reshape(x_test.shape[0], chans, samples)

model = EEGNet(nb_classes=2, Chans=chans, Samples=samples,
               dropoutRate=0.25, kernLength=64, F1=8, D=2, F2=16,
               dropoutType='Dropout')

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics = ['accuracy'])

# numParams = model.count_params()
# print(numParams)
# exit()
checkpointer = ModelCheckpoint(filepath='/tmp/ERN/checkpoint.h5', verbose=1,
                               save_best_only=True)
class_weights = {
    0: 1, 1: 4
}

fittedModel = model.fit(x_train, y_train, batch_size=34, epochs=500,
                        verbose=2, validation_data=(x_validate, y_valid),
                        callbacks=[checkpointer], class_weight=class_weights)

model.load_weights('/tmp/ERN/checkpoint.h5')

y_probs = model.predict(x_test)
fpr, tpr, thresholds = roc_curve(truth, y_probs[:,1], pos_label=1)
aucs = auc(fpr, tpr)
csv = pd.DataFrame(y_probs[:,1])
csv.to_csv('./out/A.csv', index=False)
print("Aucs Score", aucs)
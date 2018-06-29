# Introduction
This is the Army Research Laboratory (ARL) EEGModels project: A Collection of Convolutional Neural Network (CNN) models for EEG signal processing and classification, written in Keras and Tensorflow. The aim of this project is to

- provide a set of well-validated CNN models for EEG signal processing and classification
- facilitate reproducible research and
- enable other researchers to use and compare these models as easy as possible on their data

# Requirements

- Python (either 2.7 or 3.6)
- Keras >= 2.1.5
- Tensorflow >= 1.5.0
- Keras variable 'image_data_format' = "channels_first" in keras.json configuration file
- Data shape = (trials, kernels, channels, samples), which for the input layer, will be (trials, 1, channels, samples). 

# Models Implemented

- EEGNet [[1]](https://arxiv.org/abs/1611.08024). Both the original model and the revised model are implemented.
- EEGNet variant used for classification of Steady State Visual Evoked Potential (SSVEP) Signals [[2]](https://arxiv.org/abs/1803.04566)
- DeepConvNet [[3]](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730)
- ShallowConvNet [[3]](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730)


# Usage

To use this package, place the contents of this folder in your PYTHONPATH environment variable. Then, one can simply import any model and configure it as


```python

from EEGModels import EEGNet, ShallowConvNet, DeepConvNet

model  = EEGNet(nb_classes = ..., Chans = ..., Samples = ...)

model2 = ShallowConvNet(nb_classes = ..., Chans = ..., Samples = ...)

model3 = DeepConvNet(nb_classes = ..., Chans = ..., Samples = ...)

```

Compile the model with the associated loss function and optimizer (in our case, the categorical cross-entropy and Adam optimizer, respectively). Then fit the model and predict on new test data.

```python

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
fittedModel    = model.fit(...)
predicted      = model.predict(...)

```

# EEGNet Feature Explainability

To reproduce the EEGNet single-trial feature relevance results as we reported in [[1]](https://arxiv.org/abs/1611.08024), download and install DeepExplain located [[here]](https://github.com/marcoancona/DeepExplain), which implements a variety of relevance attribution methods (both gradient-based and perturbation-based). A sketch of how to use it is given below:

```python
from EEGModels import EEGNet
from keras.models import Model
from deepexplain.tensorflow import DeepExplain
import keras.backend as K

# configure, compile and fit the model
 
model  = EEGNet(nb_classes = ..., Chans = ..., Samples = ...)
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
fittedModel    = model.fit(...)

# use DeepExplain to get individual trial feature relevances for some test data (X_test, Y_test). 
# Note that model.layers[-2] points to the dense layer prior to softmax activation. Also, we use
# the DeepLIFT method in the paper, although other options, including epsilon-LRP, are available.
# This works with all implemented models. 

# here, Y_test and X_test are the one-hot encodings of the class labels. 

with DeepExplain(session = K.get_session()) as de:
	input_tensor = model.layers[0].input
	fModel = Model(inputs = input_tensor, outputs = model.layers[-2].output)    
	target_tensor = fModel(input_tensor)    

	# can use epsilon-LRP as well if you like.
	attributions = de.explain('deeplift', target_tensor * Y_test, input_tensor, X_test)
	# attributions = de.explain('elrp', target_tensor * Y_test, input_tensor, X_test)	


```


# Paper Citation

If you use the EEGNet model in your research and found it helpful, please cite the following paper:

```
@article{LawhernSWGHL16,
  author    = {Vernon J. Lawhern and
               Amelia J. Solon and
               Nicholas R. Waytowich and
               Stephen M. Gordon and
               Chou P. Hung and
               Brent J. Lance},
  title     = {EEGNet: {A} Compact Convolutional Network for EEG-based Brain-Computer
               Interfaces},
  journal   = {arXiv preprint arXiv:1611.08024},
  year      = {2016},
  url       = {http://arxiv.org/abs/1611.08024}
}
```

If you use the SSVEP variant of the EEGNet model in your research and found it helpful, please cite the following paper:

```
@article{Waytowich2018,
  author    = {Nicholas R. Waytowich and
               Vernon Lawhern and
               Javier O. Garcia and
               Jennifer Cummings and
               Josef Faller and
               Paul Sajda and
               Jean M. Vettel},
  title     = {Compact Convolutional Neural Networks for Classification of Asynchronous
               Steady-state Visual Evoked Potentials},
  journal   = {arXiv preprint arXiv:1803.04566},
  year      = {2018},
  url       = {http://arxiv.org/abs/1803.04566}
}
```

Similarly, if you use the ShallowConvNet or DeepConvNet models and found them helpful, please cite the following paper:

```
@article{hbm23730,
author = {Schirrmeister Robin Tibor and 
          Springenberg Jost Tobias and 
          Fiederer Lukas Dominique Josef and 
          Glasstetter Martin and 
          Eggensperger Katharina and 
          Tangermann Michael and 
          Hutter Frank and 
          Burgard Wolfram and 
          Ball Tonio},
title = {Deep learning with convolutional neural networks for EEG decoding and visualization},
journal = {Human Brain Mapping},
volume = {38},
number = {11},
pages = {5391-5420},
keywords = {electroencephalography, EEG analysis, machine learning, end‐to‐end learning, brain–machine interface, brain–computer interface, model interpretability, brain mapping},
doi = {10.1002/hbm.23730},
url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/hbm.23730}
}
```

# Legal Disclaimer

This project is governed by the terms of the Creative Commons Zero 1.0 Universal (CC0 1.0) Public Domain Dedication (the Agreement). You should have received a copy of the Agreement with a copy of this software. If not, see https://github.com/USArmyResearchLab/ARLDCCSO. Your use or distribution of ARL EEGModels, in both source and binary form, in whole or in part, implies your agreement to abide by the terms set forth in the Agreement in full. 
 
Other portions of this project are subject to domestic copyright protection under 17 USC Sec. 105.  Those portions are licensed under the Apache 2.0 license.  The complete text of the license governing this material is in the file labeled LICENSE.TXT that is a part of this project's official distribution. 

arl-eegmodels is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 

You may find the full license in the file LICENSE in this directory.

# Contributions

Due to legal issues, every contributor will need to have a signed Contributor License Agreement on file. The ARL Contributor License Agreement (ARL Form 266) can be found [here](https://github.com/USArmyResearchLab/ARL-Open-Source-Guidance-and-Instructions/blob/master/ARL%20Form%20-%20266.pdf). 

Each external contributor must execute and return a copy for each project that he or she intends to contribute to. 

Once ARL receives the executed form, it will remain in force permanently. 

Thus, external contributors need only execute the form once for each project that they plan on contributing to.



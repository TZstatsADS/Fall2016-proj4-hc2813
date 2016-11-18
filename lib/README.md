# Project: Words 4 Music

### Code lib Folder

The lib directory contains various files with function definitions (but only function definitions - no code that actually runs).

##1 Feature Selection
### 1.1 seletced features:
"beats_confidence", "segments_confidence", "segments_loudness_max", "segments_pitches", "segments_timbre", "tatums_confidence", and "tatums_start".
### 1.2 data processing
Due to the dimension of each data is different from each other, so this project choose the first 100 element in each vector in selected features, and drop out the data with dimension lower than 100.

##2 Model Data
### 2.1 Model Selection
Base on the intuition that comparing the similarity among different data, KNN algorithms are chosen to model the data.

##3 Testing Data
The submission.csv constains the ranks of each words in every single song, which are computed by the KNN model trained by all training data.


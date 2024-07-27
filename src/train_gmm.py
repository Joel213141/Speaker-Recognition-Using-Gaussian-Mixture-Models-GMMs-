{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d792f5e6",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pickle\n",
    "from sklearn.mixture import GaussianMixture\n",
    "import os\n",
    "\n",
    "def learningGMM(features, n_components=5, max_iter=50):\n",
    "    gmm = GaussianMixture(n_components=n_components, max_iter=max_iter)\n",
    "    gmm.fit(features)\n",
    "    return gmm\n",
    "\n",
    "def train_gmms(feature_dir, model_dir, speakers):\n",
    "    for speaker in speakers:\n",
    "        with open(f'{feature_dir}/{speaker}_mfcc.fea', 'rb') as f:\n",
    "            features = pickle.load(f)\n",
    "        gmm = learningGMM(features)\n",
    "        with open(f'{model_dir}/{speaker}.gmm', 'wb') as f:\n",
    "            pickle.dump(gmm, f)\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    feature_dir = 'TrainingFeatures'\n",
    "    model_dir = 'Models'\n",
    "    speakers = ['list', 'of', 'speaker', 'names']  # Populate with actual speaker names from the directory listing\n",
    "    train_gmms(feature_dir, model_dir, speakers)\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

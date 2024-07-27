{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "67cc8aeb",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pickle\n",
    "from extract_features import mfcc_extraction\n",
    "import os\n",
    "\n",
    "def load_gmms(model_dir):\n",
    "    gmms = {}\n",
    "    for filename in os.listdir(model_dir):\n",
    "        if filename.endswith('.gmm'):\n",
    "            with open(os.path.join(model_dir, filename), 'rb') as f:\n",
    "                gmms[filename[:-4]] = pickle.load(f)\n",
    "    return gmms\n",
    "\n",
    "def speaker_recognition(audio_file_name, gmms):\n",
    "    mfcc_features = mfcc_extraction(audio_file_name, 0.015, 12)\n",
    "    scores = {speaker: gmm.score(mfcc_features) for speaker, gmm in gmms.items()}\n",
    "    return max(scores, key=scores.get)\n",
    "\n",
    "def test_speaker_recognition(test_dir, gmms):\n",
    "    correct_predictions = 0\n",
    "    total_samples = 0\n",
    "    for root, _, files in os.walk(test_dir):\n",
    "        for file in files:\n",
    "            if file.endswith('.wav'):\n",
    "                total_samples += 1\n",
    "                audio_file_path = os.path.join(root, file)\n",
    "                predicted_speaker = speaker_recognition(audio_file_path, gmms)\n",
    "                true_speaker = os.path.basename(root)\n",
    "                if predicted_speaker == true_speaker:\n",
    "                    correct_predictions += 1\n",
    "    accuracy = (correct_predictions / total_samples) * 100\n",
    "    print(f'Recognition Accuracy: {accuracy:.2f}%')\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    model_dir = 'Models'\n",
    "    test_dir = 'SpeakerData/Test'\n",
    "    gmms = load_gmms(model_dir)\n",
    "    test_speaker_recognition(test_dir, gmms)\n"
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

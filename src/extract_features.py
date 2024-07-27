{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4ef04250",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import librosa\n",
    "from pydub import AudioSegment\n",
    "import pickle\n",
    "import os\n",
    "\n",
    "def mfcc_extraction(audio_filename, hop_duration, num_mfcc):\n",
    "    speech = AudioSegment.from_wav(audio_filename)\n",
    "    samples = speech.get_array_of_samples()\n",
    "    sampling_rate = speech.frame_rate\n",
    "    mfcc = librosa.feature.mfcc(y=np.float32(samples), sr=sampling_rate,\n",
    "                                hop_length=int(sampling_rate * hop_duration), n_mfcc=num_mfcc)\n",
    "    return mfcc.T\n",
    "\n",
    "def extract_features_for_all_speakers(base_path, speakers, hop_duration=0.015, num_mfcc=12):\n",
    "    mfcc_all_speakers = []\n",
    "    for s in speakers:\n",
    "        sub_path = os.path.join(base_path, 'Train', s)\n",
    "        sub_file_names = [os.path.join(sub_path, f) for f in os.listdir(sub_path)]\n",
    "        mfcc_one_speaker = np.asarray(())\n",
    "        for fn in sub_file_names:\n",
    "            mfcc_features = mfcc_extraction(fn, hop_duration, num_mfcc)\n",
    "            if mfcc_one_speaker.size == 0:\n",
    "                mfcc_one_speaker = mfcc_features\n",
    "            else:\n",
    "                mfcc_one_speaker = np.vstack((mfcc_one_speaker, mfcc_features))\n",
    "        mfcc_all_speakers.append(mfcc_one_speaker)\n",
    "        with open(f'TrainingFeatures/{s}_mfcc.fea', 'wb') as f:\n",
    "            pickle.dump(mfcc_one_speaker, f)\n",
    "    return mfcc_all_speakers\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    path = 'C:/Users/User/Desktop/Class Folder/Computer Vision/SpeakerData/'\n",
    "    speakers = os.listdir(path + 'Train/')\n",
    "    extract_features_for_all_speakers(path, speakers)\n"
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

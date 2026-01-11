#!/bin/bash

cd data
mkdir binary_classification
cd binary_classification

# ParlVote: A Corpus for Sentiment Analysis of Political Debates
mkdir ParlVote
cd ParlVote
wget https://data.mendeley.com/public-files/datasets/czjfwgs9tm/files/8f835544-9c55-40dc-b91f-829b8cb7c80c/file_downloaded
cd ..

# 'Aye' or 'No'? Speech-level Sentiment Analysis of Hansard UK Parliamentary Debate Transcripts
mkdir HanDeSeT
cd HanDeSeT
wget https://data.mendeley.com/public-files/datasets/xsvp45cbt4/files/288aaace-8dfd-49e4-9140-7cf79939f053/file_downloaded
cd ..

# Stance classification: a comparative study and use case on Australian parliamentary debates
mkdir AusHansard
cd AusHansard
wget https://github.com/stephanienzz/stance-classification/raw/refs/heads/main/data/aus_augmented_features.csv.zip
wget https://github.com/stephanienzz/stance-classification/raw/refs/heads/main/data/handeset_features.csv.zip
wget https://github.com/stephanienzz/stance-classification/raw/refs/heads/main/data/parlvote_features.csv.zip
unzip aus_augmented_features.csv.zip
unzip handeset_features.csv.zip
unzip parlvote_features.csv.zip
rm *.zip
cd ..

# Get out the vote: Determining support or opposition from congressional floor-debate transcripts.
mkdir ConVote
cd ConVote
wget https://www.cs.cornell.edu/home/llee/data/convote/convote_v1.1.tar.gz
tar -xvzf convote_v1.1.tar.gz
rm convote_v1.1.tar.gz
cd ..

cd ..
mkdir multi_class_classification
cd multi_class_classification
mkdir ArgUNSC

# Policy-focused Stance Detection in Parliamentary Debate Speeches
mkdir ParlVote+
cd ParlVote+
wget --content-disposition "https://drive.google.com/uc?export=download&id=1_pNE8N-shWgKfoQEhMNdingnXiuYZ79K"
cd ..


# Policy Preference Detection in Parliamentary Debate Motions.
mkdir MotionPolicyPreference
cd MotionPolicyPreference
wget https://madata.bib.uni-mannheim.de/308/1/MotionPolicyPreferences%20-%20Gold.csv
cd ..

# Argument-based detection and classification of fallacies in political debates
mkdir ElecDeb60to20-relations
cd ElecDeb60to20-relations
wget https://raw.githubusercontent.com/deborahdore/ElecDeb60to20/refs/heads/main/data/relations/dev.csv
wget https://raw.githubusercontent.com/deborahdore/ElecDeb60to20/refs/heads/main/data/relations/train.csv
wget https://raw.githubusercontent.com/deborahdore/ElecDeb60to20/refs/heads/main/data/relations/test.csv
cd ..

cd ..
mkdir sequence_labelling
cd sequence_labelling

# Argument-based detection and classification of fallacies in political debates
mkdir ElecDeb60to20-components
cd ElecDeb60to20-components
wget https://raw.githubusercontent.com/deborahdore/ElecDeb60to20/refs/heads/main/data/components/dev.conll
wget https://raw.githubusercontent.com/deborahdore/ElecDeb60to20/refs/heads/main/data/components/train.conll
wget https://raw.githubusercontent.com/deborahdore/ElecDeb60to20/refs/heads/main/data/components/test.conll
cd ..

# From Debates to Diplomacy: Argument Mining Across Political Registers
mkdir ArgUNSC
cd ArgUNSC
wget https://raw.githubusercontent.com/mpoiaganova/political-argument-mining/refs/heads/main/data/base.csv
cd ..

cd ..
mkdir ner
cd ner
mkdir nerex
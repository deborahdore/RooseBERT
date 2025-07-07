#!/bin/bash

cd data
rm -rf relation_classification
mkdir relation_classification

rm -rf argument_detection
mkdir argument_detection

rm -rf sentiment_analysis
mkdir sentiment_analysis

rm -rf ner
mkdir ner


cd relation_classification
wget https://raw.githubusercontent.com/pierpaologoffredo/ElecDeb60to20/refs/heads/main/data/relation_data/latest_test.tsv
wget https://raw.githubusercontent.com/pierpaologoffredo/ElecDeb60to20/refs/heads/main/data/relation_data/latest_train.tsv
wget https://raw.githubusercontent.com/pierpaologoffredo/ElecDeb60to20/refs/heads/main/data/relation_data/latest_dev.tsv

cd ..
cd argument_detection
wget https://raw.githubusercontent.com/crscardellino/argumentation-mining-transformers/refs/heads/master/data/sequence/disputool-test.conll
wget https://raw.githubusercontent.com/crscardellino/argumentation-mining-transformers/refs/heads/master/data/sequence/disputool-train.conll
wget https://raw.githubusercontent.com/crscardellino/argumentation-mining-transformers/refs/heads/master/data/sequence/disputool-validation.conll

cd ..
cd sentiment_analysis
wget https://data.mendeley.com/public-files/datasets/czjfwgs9tm/files/8f835544-9c55-40dc-b91f-829b8cb7c80c/file_downloaded

cd ..
cd ner
wget https://raw.githubusercontent.com/zliucr/CrossNER/refs/heads/main/ner_data/politics/dev.txt
wget https://raw.githubusercontent.com/zliucr/CrossNER/refs/heads/main/ner_data/politics/train.txt
wget https://raw.githubusercontent.com/zliucr/CrossNER/refs/heads/main/ner_data/politics/test.txt
#!/bin/bash

########################################## TRAINING DATASET ##################################################
# Manually download all of these files and then start this script

# https://zenodo.org/records/8121950/files/hansard-corpus.zip?download=1
# https://dataverse.harvard.edu/file.xhtml?persistentId=doi:10.7910/DVN/XPCVEI/ORRNK4&version=3.0
# https://dataverse.harvard.edu/file.xhtml?fileId=6435506&version=2.0
# https://reshare.ukdataservice.ac.uk/854292/
# https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/0TJX8Y
# https://dataverse.harvard.edu/file.xhtml?fileId=10809805&version=6.1
# https://dataverse.harvard.edu/file.xhtml?fileId=4432885&version=1.0
# https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6MZN76
# https://www.clarin.si/repository/xmlui/handle/11356/2006
# https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/HISX4G
# https://dataverse.harvard.edu/file.xhtml?persistentId=doi:10.7910/DVN/L4OAKN/LLMYON&version=1.0

echo "scraping_africa"
python -m script.scraping.scraping_africa

echo "scraping_australia"
python -m script.scraping.scraping_australia

echo "scraping_europarl"
python -m script.scraping.scraping_europarl

echo "scraping_scotland"
python -m script.scraping.scraping_scotland

echo "scraping_uk"
python -m script.scraping.scraping_uk

echo "scraping_ungdc"
python -m script.scraping.scraping_ungdc

echo "scraping_ungdc"
python -m script.scraping.scraping_unsc

echo "scraping_us"
python -m script.scraping.scraping_us

echo "scraping_new_zealand"
python -m script.scraping.scraping_new_zealand

echo "scraping_ireland"
python -m script.scraping.scraping_ireland

echo "scraping_canada"
python -m script.scraping.scraping_canada

#echo "scraping_translated"
#python -m script.scraping.scraping_translated

echo "preparing training dataset"
python -m script.prepare_training_dataset

########################################################################################################################
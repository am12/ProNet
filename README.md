# ProNet

a CNN deep learning model for protein function prediction

## Download instructions

Go to https://ftp.ncbi.nlm.nih.gov/pub/COG/COG2020/data/

Download all the files at the link to the data/ directory:
- cog-20.cog.csv
- cog-20.def.tab
- cog-20.fa.gz
- cog-20.org.csv
- cog-20.patt.txt
- cog-20.tax.csv
- fun-20.tab.txt

Make sure to expand the cog-20.fa.gz file. 

## Nice to Have

1. Experiment with different embedding/representations of data
https://pubmed.ncbi.nlm.nih.gov/33143605/ -> variable length n-gram embedding
https://academic.oup.com/bioinformatics/article/34/15/2642/4951834 -> learned embeddings
https://academic.oup.com/bioinformatics/article/40/1/btad786/7510842 -> embedding-based alignment

## Todo

1. Format the data into valid input for CNN model
    - one-hot encoding
    - padding so equal lengths
    - check to see whether 3'-5' order

2. Create model architecture
    - adapt from Splam model

3. Train on sample data until overfit
    - show that the pattern can be learned on small datasets

4. Data pre-processing pipeline
    - create train, test, val splits
    - for each get the sequence and label 
    - create pipeline for compiling datasets (maybe put into DataLoader)

5. Training pipeline
    - train the model on the input data with labels 

6. Validation pipeline
    - test model on validation sets
    - develop pipeline to calculate confusion matrix, PR, ROC statistics of classification
    - show k-fold validation statistics

7. Fine-tune hyperparameter space of model
    - tune the hyperparameters of model until achieving high validation accuracy
    - play around with loss function, optimizer, and gradient descent method

8. Test pipeline
    - test the model on final dataset to evaluate performance
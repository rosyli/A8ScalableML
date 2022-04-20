#!/usr/bin/env python3
import cloudpickle, sys
import fsspec
from rdkit.Chem import AllChem
from rdkit import Chem
import pandas as pd
from IPython.display import display

#fetch and load model from Google Cloud
model = cloudpickle.load(fsspec.open_files("gs://assign6_lrx/rf_100tree_0.33ft.pkl","rb")[0].open())
# model = cloudpickle.load(
#     open("/net/dali/home/mscbio/rul49/assign8/rf_100tree_0.33ft.pkl", "rb"))
# infile = open(sys.argv[1])
# out = open(sys.argv[2],'wt')
# infile = open("/net/dali/home/mscbio/rul49/assign8/test1.txt")

# infile = pd.read_csv("/net/dali/home/mscbio/rul49/assign8/test1.txt", sep=" ")
infile = pd.read_csv(sys.argv[1])

# create ecfp4 dataframe (compounds)
fps = [
    list(
        AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(i),
                                              2,
                                              nBits=1024).ToBitString())
    for i in infile["SMILES"]
]
df_ecfp4 = pd.DataFrame(fps,
                        columns=["ECFP4." + str(i) for i in range(1, 1025)])

# create pssm dataframe (kinase)
# k_sep_norm2 = pd.read_csv("./assign8/Heval_model/k-sep_normalized.tsv",
#                           sep="\t")
k_sep_norm2 = pd.read_csv("gs://assign6_lrx/k-sep_normalized.tsv",
                          sep="\t")

k_sep_norm2.rename(columns={'target_id': "UniProt"}, inplace=True)

# merge k_sep_norm2 into infile
infile_1 = infile.merge(k_sep_norm2, on="UniProt")

# merge df_ecfp4 into infile
infile_2 = pd.concat([infile_1, df_ecfp4], axis=1)

X_test = infile_2.iloc[:, 2:]

prediction = model.predict(X_test)  # array

i = 0
# infile_3 = open("/net/dali/home/mscbio/rul49/assign8/test1.txt")
# out = open("/net/dali/home/mscbio/rul49/assign8/test1_prediction.txt", 'wt')

infile_3 = open(sys.argv[1])
out = open(sys.argv[2],'wt')

out.write(infile_3.readline())

for line in infile_3:
    smile, uniprot = line.strip().split()

    val = prediction[i]
    i = i + 1

    out.write(f'{smile} {uniprot} {val:.4f}\n')

# # out.write(infile.readline()) # header
# i = 0
# for line in infile:
#     if i > 0:
#         smile, uniprot = line.strip().split()

#         # convert smile string to ecfp4 vector
#         smile_ecfp4 = AllChem.GetMorganFingerprintAsBitVect(
#             Chem.MolFromSmiles(smile), 2, nBits=1024).ToBitString()

#         # convert uniprot id to pssm vector

#         val = model.predict((smile, uniprot))
#     i = i + 1
#     # out.write(f'{smile} {uniprot} {val:.4f}\n')
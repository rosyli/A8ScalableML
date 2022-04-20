#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import argparse
import cloudpickle # cloudpickle provides the most robust saving of objects
import pickle
import dask.dataframe as dd
from sklearn.ensemble import RandomForestRegressor
from rdkit.Chem import AllChem
from rdkit import Chem

#protein and compound feature vector file
k_sep_norm = dd.read_csv("./assign8/Heval_model/k-sep_normalized.tsv",sep="\t")  #protein
ecfp4 = dd.read_csv("./assign8/Heval_model/ecfp4.tsv",sep="\t")  #compound

#train dataset
df_tr = dd.read_csv("./assign8/Heval_model/train_set_87989.tsv",sep="\t")
df_vl = dd.read_csv("./assign8/Heval_model/validation_set_6195.tsv",sep="\t")

#train dataset with protein-compound feature vectors
df_tr_ft = df_tr.merge(k_sep_norm,on="target_id").merge(ecfp4,on="compound_id").compute()
df_vl_ft = df_vl.merge(k_sep_norm,on="target_id").merge(ecfp4,on="compound_id").compute()

#features and labels of train dataset
X_train = pd.concat([df_tr_ft.iloc[:,3:],df_vl_ft.iloc[:,3:]])
y_train = pd.concat([df_tr_ft["pchembl_value"],df_vl_ft["pchembl_value"]])

#round2 input file
df_test = pd.read_csv("./assign8/Heval_model/input.csv")
print("files loaded!")

#round2 input file with protein-compound feature vectors
k_sep_norm2 = pd.read_csv("./assign8/Heval_model/k-sep_normalized.tsv",sep="\t")
k_sep_norm2.rename(columns={'target_id':"UniProt_Id"},inplace=True)

df_test_k_sep = df_test.merge(k_sep_norm2,on="UniProt_Id")

fps = [list(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(i),2,nBits=1024).ToBitString()) for i in df_test_k_sep["Compound_SMILES"]]

df_ecfp4 = pd.DataFrame(fps, columns=["ECFP4."+str(i) for i in range(1,1025)])
df_test_ft = pd.concat([df_test_k_sep,df_ecfp4],axis=1)

X_test = df_test_ft.iloc[:,6:]
print("features loaded!")

#RF regression based model generation and prediction
RFreg = RandomForestRegressor(n_estimators=100,max_features=0.33,random_state=42) 
RFreg.fit(X_train, y_train)
pred_result = RFreg.predict(X_test)
print("model generated!")

#adding prediction results to the input file and saving the file as output 
df_test.insert(loc=6,column="pKd_[M]_pred",value=pred_result)  
df_test.to_csv("./assign8/Heval_model/prediction.csv",index=None)

#saving the generated model  
with open("rf_100tree_0.33ft.pkl","wb") as file:
    cloudpickle.dump(RFreg,file)

print("done!")

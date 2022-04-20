# -*- coding: utf-8 -*-
import pandas as pd
import pickle
from rdkit.Chem import AllChem
from rdkit import Chem

#round2 input file
df_test = pd.read_csv(r"input.csv")  

#round2 input file with protein-compound feature vectors
k_sep_norm = pd.read_csv(r"k-sep_normalized.tsv",sep="\t")
k_sep_norm.rename(columns={'target_id':"UniProt_Id"},inplace=True)

df_test_k_sep = df_test.merge(k_sep_norm,on="UniProt_Id")

fps = [list(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(i),2,nBits=1024).ToBitString()) for i in df_test_k_sep["Compound_SMILES"]]

df_ecfp4 = pd.DataFrame(fps, columns=["ECFP4."+str(i) for i in range(1,1025)])
df_test_ft = pd.concat([df_test_k_sep,df_ecfp4],axis=1)

X_test = df_test_ft.iloc[:,6:]

#model prediction for round2 input file  
with open(r"0.3std_filtered_round2-proteins_k-sep_no-thr_rf_100tree_0.33ft.pkl", 'rb') as f:  
    pickle_model = pickle.load(f)    
    pred_result = pickle_model.predict(X_test)

#adding prediction results to the input file and saving the file as output 
df_test.insert(loc=6,column="pKd_[M]_pred",value=pred_result)  
df_test.to_csv(r"prediction.csv",index=None)

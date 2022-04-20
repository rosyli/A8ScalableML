#!/usr/bin/env python3
import cloudpickle, sys
import fsspec

#fetch and load model from Google Cloud
# model = cloudpickle.load(fsspec.open_files("gs://assign8rl/randomforest_100tree_0.33ft.pkl","rb")[0].open())
model = cloudpickle.load(open("/net/dali/home/mscbio/rul49/assign8/rf_100tree_0.33ft.pkl","rb"))
# infile = open(sys.argv[1])
# out = open(sys.argv[2],'wt')
infile = open("/net/dali/home/mscbio/rul49/assign8/test1.txt")

# out.write(infile.readline()) # header
i = 0
for line in infile:
    if i > 0:
        smile, uniprot = line.strip().split()
        val = model.predict((smile, uniprot))
    i = i + 1
    # out.write(f'{smile} {uniprot} {val:.4f}\n')
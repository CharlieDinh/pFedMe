# MNIST Dataset



To generate non-iid data:

```
mkdir test
mkdir train
python generate_niid.py
```

The layout of the folders under `./mnist` should be:

```
| data

----| mldata

---- ----| raw_data.mat

----| train 

---- ----| train_file_name.json

----| test

---- ----| test_file_name.json

| generate_niid.py
| README.md
```




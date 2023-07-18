# D2Co
The source code of D2Co submission

Uncovering User Interest from Biased and Noised Watch Time in Video Recommendation

Running Environment

`Python 3.8.3` `Pytorch 1.6.0`

Run the following command to preprocess the dataset

```Bash
python prepare_data.py --group_num ${group_num} --windows_size ${windows_size} --alpha ${alpha} --dat_name ${dataname} --is_load 0
```

Run the following command to train different model with different label

```Bash
python main.py --fout ../rec_datasets/results/${modelname}_${labelname}_${windows_size}_${alpha} --dat_name ${dataname} --model_name ${modelname} --label_name ${labelname} --windows_size ${windows_size} --alpha ${alpha}
```

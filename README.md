# Step 1 

Extract features from the data files.

```python
python CompileStreamsAndFeatures.py --accent_v_thresh 0.75
``` 

Then, create train/test/validation subsets

```python
python prepare_subsets.py --accent_v_thresh 0.75 --input_dir "with_features" --output_dir "model_ready"
```

# Step 2

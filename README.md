# Step 1 

Extract features from the data files.

```python
python CompileStreamsAndFeatures.py --accent_v_thresh 0.75
``` 

Then, create train/test/validation subsets

```python
python SubsetData.py --accent_v_thresh 0.75
```

# Step 2

Train using 
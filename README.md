# Step 1 

Extract features from the data files.

```python
python CompileStreamsAndFeatures.py --accent_v_thresh 0.75
``` 

Then, binning the features:

```python
python BinnifyCompiledStreams.py --accent_v_thresh 0.75 --n_bins 10 --low_percentile 0.0 --high_percentile 0.9
```


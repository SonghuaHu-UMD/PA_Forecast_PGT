# Demo: Comparison of popular temporal graph neural networks in population flow forecasting

## Environment
We use the torch >= 1.10.2 and Python 3 for implementation.

We follow the framework of [pytorch_geometric_temporal](https://github.com/SonghuaHu-UMD/pytorch_geometric_temporal) to prepare data and run the model.

Please execute the following command to get the source code.

```bash
git clone https://github.com/SonghuaHu-UMD/PA_Forecast_PGT
cd PA_Forecast_PGT
```

## Data Preparation
The datasets are private. Contact me to get the data. Unzip the data and put it into the `data` folder.


## Model Training
The script `1.2-Model_multime.py` is used for training and evaluating the main model.

The script `1.3-Model_Best_Result_Output.py` is used to transform the best prediction output.

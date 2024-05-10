import pandas as pd
from tqdm import tqdm
import numpy as np
import time
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA, IncrementalPCA as IPCA
import plotly.express as px
import gc
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score
import h5py
import os
from joblib import dump, load


pd.set_option('io.hdf.default_format', 'table')


def rearrange_df(data, leave_out_cols):
    leave_out_df = pd.concat([data.pop(col) for col in leave_out_cols], axis=1)
    data = pd.concat([data, leave_out_df], axis=1)
    return data


TRAIN = './../data/train.csv'
TEST = './../data/test.csv'
SUB = './../data/sample_submission.csv'


X_LEAVE_OUT_COLS = ['pbuf_SOLIN', 'pbuf_SHFLX', 'pbuf_LHFLX', 'cam_in_LWUP', 'state_ps']
Y_LEAVE_OUT_COLS = ['cam_out_SOLLD', 'cam_out_SOLL', 'cam_out_SOLS', 'cam_out_SOLSD', 'cam_out_FLWDS', 'cam_out_NETSW']
X_COLS = rearrange_df(pd.read_csv(TEST, nrows=1).drop('sample_id', axis=1), X_LEAVE_OUT_COLS).columns.tolist()
Y_COLS = rearrange_df(pd.read_csv(SUB, nrows=1).drop('sample_id', axis=1), Y_LEAVE_OUT_COLS).columns.tolist()

# For debugging notebook, change to 500_000 before running notebook
CHECK_ROWS = 500_000

start = time.time()

# For example, let's consider 500,000 rows
chunk = pd.read_csv(TRAIN, nrows=CHECK_ROWS, usecols=X_COLS+Y_COLS)
original_mem_used = chunk.memory_usage(deep=True).sum() / (1024*1024)
print(f'Memory used by the original chunk: {original_mem_used} MB')

chunk_32 = chunk.astype('float32')
f32_mem_used = chunk_32.memory_usage(deep=True).sum() / (1024*1024)
print(f'Memory used by the downcasted chunk: {f32_mem_used} MB')
print('-'*80)

print('Mean MSE over all features between original and downcasted Dataset: ', ((chunk - chunk_32) ** 2).mean().mean())

print('Max MSE over all data values between original and downcasted Dataset: ', ((chunk - chunk_32) ** 2).max().max())
print('-'*80)

print(f"Took {time.time() - start}s to finish this cell")
print(f'Total Memory usage reduced from original: {original_mem_used/f32_mem_used}x')


class CustomPipeline:
    def __init__(self, all_cols, leave_out_cols, pca_components, batch_size):
        self.cols = all_cols
        self.scaler = StandardScaler()
        self.leave_out_cols = leave_out_cols
        self.pca = IPCA(n_components=pca_components, batch_size=batch_size)

    def partial_fit(self, data):
        self.scaler.partial_fit(data)
        temp_data = pd.DataFrame(self.scaler.transform(data),
                                 columns=self.cols).astype('float32')
        temp_data.drop(self.leave_out_cols, axis=1, inplace=True)
        self.pca.partial_fit(temp_data.to_numpy())
        gc.collect()

    def transform(self, data):
        temp_data = pd.DataFrame(self.scaler.transform(data).astype('float32'),
                                 columns=self.cols)
        backup_cols = temp_data[self.leave_out_cols]
        temp_data.drop(self.leave_out_cols, axis=1, inplace=True)
        temp_data = self.pca.transform(temp_data.to_numpy())
        return np.hstack((temp_data, backup_cols.to_numpy())).astype('float32')

    def inverse_transform(self, data):
        backup_cols = data[:, -len(self.leave_out_cols):].copy()
        data = data[:, :-len(self.leave_out_cols)].copy()
        data = self.pca.inverse_transform(data)
        data = pd.DataFrame(np.hstack((data, backup_cols)), columns=self.cols)
        return self.scaler.inverse_transform(data).astype('float32')


start = time.time()

chunk_x = chunk_32[X_COLS].copy(deep=True)
chunk_y = chunk_32[Y_COLS].copy(deep=True)

pipeline_x = CustomPipeline(X_COLS, X_LEAVE_OUT_COLS, 200, CHECK_ROWS)
pipeline_y = CustomPipeline(Y_COLS, Y_LEAVE_OUT_COLS, 270, CHECK_ROWS)

pipeline_x.partial_fit(chunk_x.copy())
pipeline_y.partial_fit(chunk_y.copy())

transformed_x = pd.DataFrame(pipeline_x.transform(chunk_x)).astype('float32')
gc.collect()
transformed_y = pd.DataFrame(pipeline_y.transform(chunk_y)).astype('float32')
gc.collect()

x_pca_components = pipeline_x.pca.explained_variance_ratio_
y_pca_components = pipeline_y.pca.explained_variance_ratio_

print(f'PCA components to preserve {x_pca_components.sum() * 100}% variance on predictors: {len(x_pca_components)}')
print(f'PCA components to preserve {y_pca_components.sum() * 100}% variance on targets: {len(y_pca_components)}')
print('-' * 80)

transformedx_mem_used = transformed_x.memory_usage(deep=True).sum() / (1024 * 1024)
transformedy_mem_used = transformed_y.memory_usage(deep=True).sum() / (1024 * 1024)

print(f'Memory used by the original Data: {original_mem_used} MB')
print(f'Memory used by the transformed Data: {transformedx_mem_used + transformedy_mem_used} MB')
print(f'Total Memory usage reduced from original: {original_mem_used / (transformedx_mem_used + transformedy_mem_used)}x')
print('-' * 80)

invtransformed_x = pipeline_x.inverse_transform(transformed_x.to_numpy())
invtransformed_y = pipeline_y.inverse_transform(transformed_y.to_numpy())

gc.collect()

fig = px.line(((invtransformed_y - chunk_32[Y_COLS]) ** 2).mean())
fig.show()

print(f"Took {time.time() - start}s to finish this cell")
print(f'Total Memory usage reduced from original: {original_mem_used / (transformedx_mem_used + transformedy_mem_used)}x')

total_data = pd.concat((transformed_x, transformed_y), axis=1).astype('float32')
final_shape = total_data.shape


def get_score_from_data(data_x, data_y, pipeline):
    """

    :param data_x:
    :param data_y:
    :param pipeline:
    :return:
    """
    start = time.time()
    cv = KFold(n_splits=3, random_state=42, shuffle=True)
    model = KNeighborsRegressor(n_neighbors=30)
    for train_index, test_index in cv.split(data_x):
        X_train, X_test = data_x.iloc[train_index], data_x.iloc[test_index]
        y_train, y_test = data_y.iloc[train_index], data_y.iloc[test_index]
        chunk_y_test = chunk_y.iloc[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # If pipeline is not provided, then its original data, else its transformed
        # and needs inverse transforming.
        if pipeline is not None:
            y_pred = pipeline.inverse_transform(y_pred)

        scores = r2_score(chunk_y_test, y_pred)
        print("Average over all Columns R2 score this split: ",
              np.mean(scores))
        print('-' * 80)

    print(f"Total Time Taken: {time.time() - start} ")


print('-' * 80)
print("Fitting on KNeighborsRegressor on the transformed data:")
get_score_from_data(transformed_x, transformed_y, pipeline_y)

_ = gc.collect()

get_score_from_data(chunk_x, chunk_y, None)


def get_entire_train_data_chunks():
    """

    :return:
    """
    reader = pd.read_csv(TRAIN, chunksize=CHECK_ROWS)
    for chunk in reader:
        chunk = chunk[X_COLS + Y_COLS].astype('float32')
        yield chunk


del chunk, chunk_32, chunk_x, chunk_y, transformed_x, transformed_y, total_data, invtransformed_x, invtransformed_y
_ = gc.collect()

fpipeline_x = CustomPipeline(X_COLS, X_LEAVE_OUT_COLS, 200, CHECK_ROWS)
fpipeline_y = CustomPipeline(Y_COLS, Y_LEAVE_OUT_COLS, 270, CHECK_ROWS)

counter = 0

for chunk in get_entire_train_data_chunks():
    _ = gc.collect()
    start = time.time()

    chunk_x = chunk[X_COLS].copy(deep=True)
    chunk_y = chunk[Y_COLS].copy(deep=True)

    fpipeline_x.partial_fit(chunk_x)
    fpipeline_y.partial_fit(chunk_y)

    print(f'Finished {counter + 1} chunk in {time.time() - start} seconds.')
    counter += 1


# saving the fitted Pipelines to inverse transform later when you need it.
dump(fpipeline_x, './../models/final_pipelinex.joblib')
dump(fpipeline_y, './../models/final_pipeliney.joblib')


h5_filename = './../data/train_data_transformed.h5'
with h5py.File(h5_filename, 'w') as f:
    dset = f.create_dataset('train_transformed', (0, final_shape[1]),
                            maxshape=(None, final_shape[1]), dtype='float32',
                            compression='gzip', compression_opts=9)

counter = 0
# Now to transform and save the training data.
for chunk in get_entire_train_data_chunks():
    _ = gc.collect()
    start = time.time()

    chunk_x = chunk[X_COLS].copy(deep=True)
    chunk_y = chunk[Y_COLS].copy(deep=True)

    transformed_x = fpipeline_x.transform(chunk_x)
    transformed_y = fpipeline_y.transform(chunk_y)

    total_data = np.concatenate((transformed_x, transformed_y), axis=1).astype('float32')
    with h5py.File(h5_filename, 'a') as f:
        dset = f['train_transformed']
        new_size = dset.shape[0] + total_data.shape[0]
        dset.resize(new_size, axis=0)
        dset[dset.shape[0] - total_data.shape[0]:] = total_data
        f.flush()
        print(f'File size and shape after append: {os.path.getsize(h5_filename) / (1024 * 1024)} MB and {dset.shape}')

    print(f'Finished chunk {counter + 1} in {time.time() - start} seconds.')
    counter += 1
    _ = gc.collect()

import pandas as pd
from preprocess import LogScaleTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

# df = pd.read_csv("log2.csv", header=0)
# y = df[["Action"]]
# df = df.drop(columns="Action")

num_cols = ['Bytes', 'Bytes Sent',
       'Bytes Received', 'Packets', 'Elapsed Time (sec)', 'pkts_sent',
       'pkts_received']


def categorize_port(port, header):
    if port <= 1023:
        return f'{header}-well-known'
    elif port <= 49151:
        return f'{header}-registered'
    else:
        return f'{header}-dynamic'
    
def categorize_ports(df):
    df['Source Port'] = df['Source Port'].apply(lambda x: categorize_port(x, header="source-port"))
    df['Destination Port'] = df['Destination Port'].apply(lambda x: categorize_port(x, header="destination-port"))
    df['NAT Source Port'] = df['NAT Source Port'].apply(lambda x: categorize_port(x, header="nat-source-port"))
    df['NAT Destination Port'] = df['NAT Destination Port'].apply(lambda x: categorize_port(x, header="nat-destination-port"))
    return df

categorical_features = ['Source Port', 'Destination Port', 'NAT Source Port',
       'NAT Destination Port']

preprocessor = ColumnTransformer(
    transformers=[
        ("onehot_action", OneHotEncoder(sparse_output=False), categorical_features),
    ],
    remainder="passthrough"
)

# Create a pipeline for preprocessing
preprocessing_pipeline = Pipeline(steps=[
    ('log_scale', LogScaleTransformer(numeric_cols=num_cols)),
    ('categorize_ports', FunctionTransformer(categorize_ports, validate=False)),
    ('preprocessing', preprocessor),
    
])

        
import torch
from model import SmallNet
import pandas as pd
import joblib

def predict(model, input_data, preprocessing_pipeline):
    model.eval()
    input_data_transformed = preprocessing_pipeline.transform(input_data)
    
    input_tensor = torch.tensor(input_data_transformed, dtype=torch.float32)
    
    # Make predictions
    with torch.no_grad():
        predictions = model(input_tensor)
    

    return torch.argmax(predictions).numpy()

model = SmallNet()
model.load_state_dict(torch.load(r"trained_weights.pth", weights_only=True))

# Testing with sample data
input_data = pd.DataFrame([[24000,12000,0,0,10,6,4,40,3,20,20]], columns=['Source Port', 'Destination Port', 'NAT Source Port',
       'NAT Destination Port', 'Bytes', 'Bytes Sent', 'Bytes Received',
       'Packets', 'Elapsed Time (sec)', 'pkts_sent', 'pkts_received'])

pipeline = joblib.load("my_pipeline.pkl")
predictions = predict(model, input_data, pipeline)

print(predictions)
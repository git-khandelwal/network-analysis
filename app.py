from flask import Flask, request, render_template, render_template_string
from model import SmallNet
import pandas as pd
import torch
import joblib
from predict import predict

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            source_port = int(request.form["source_port"])
            destination_port = int(request.form["destination_port"])
            nat_source_port = int(request.form["nat_source_port"])
            nat_destination_port = int(request.form["nat_destination_port"])
            bytes_ = int(request.form["bytes_"])
            bytes_sent = int(request.form["bytes_sent"])
            bytes_received = int(request.form["bytes_received"])
            packets = int(request.form["packets"])
            elapsed_time = int(request.form["elapsed_time"])
            pkts_sent = int(request.form["pkts_sent"])
            pkts_received = int(request.form["pkts_received"])

            model = SmallNet()
            model.load_state_dict(torch.load(r"trained_weights.pth", weights_only=True))

            input_data = pd.DataFrame([[source_port,destination_port,nat_source_port,nat_destination_port,bytes_,bytes_sent,bytes_received,packets,elapsed_time,pkts_sent,pkts_received]], columns=['Source Port', 'Destination Port', 'NAT Source Port',
       'NAT Destination Port', 'Bytes', 'Bytes Sent', 'Bytes Received',
       'Packets', 'Elapsed Time (sec)', 'pkts_sent', 'pkts_received'])


            # Call the predict function
            pipeline = joblib.load("my_pipeline.pkl")
            predictions = predict(model, input_data, pipeline)
            # print(predictions)
            preds = {0: 'allow', 1: 'deny', 2: 'drop', 3: 'reset-both'}

            success_message = f"""
            <p>Successfully received the following data:</p>
            <ul>
                <li>Source Port: {source_port}</li>
                <li>Destination Port: {destination_port}</li>
                <li>NAT Source Port: {nat_source_port}</li>
                <li>NAT Destination Port: {nat_destination_port}</li>
                <li>Bytes: {bytes_}</li>
                <li>Bytes Sent: {bytes_sent}</li>
                <li>Bytes Received: {bytes_received}</li>
                <li>Packets: {packets}</li>
                <li>Elapsed Time (sec): {elapsed_time}</li>
                <li>pkts_sent: {pkts_sent}</li>
                <li>pkts_received: {pkts_received}</li>
                <li>class: {preds[int(predictions)]}</li>
            </ul>
            """
            return success_message

        except ValueError:
            return render_template("form.html", error="Please enter valid integers for all fields.")
    
    return render_template("form.html", error=None)


if __name__ == "__main__":
    app.run(debug=True)

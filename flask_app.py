from flask import Flask, request, jsonify
import grpc
import mlservice_pb2
import mlservice_pb2_grpc

app = Flask(__name__)

# Connect to gRPC server
channel = grpc.insecure_channel("localhost:50051")
stub = mlservice_pb2_grpc.ModelServiceStub(channel)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = data.get("features", [])
    
    if not features:
        return jsonify({"error": "No features provided"}), 400
    
    grpc_request = mlservice_pb2.PredictRequest(features=features)
    grpc_response = stub.Predict(grpc_request)
    
    return jsonify({"prediction": grpc_response.label})

if __name__ == "__main__":
    app.run(port=8000, debug=True)

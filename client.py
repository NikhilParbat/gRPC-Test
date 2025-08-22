import grpc
import mlservice_pb2
import mlservice_pb2_grpc

def run():
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = mlservice_pb2_grpc.ModelServiceStub(channel)
        # Example: Iris input features
        request = mlservice_pb2.PredictRequest(features=[5.1, 3.5, 1.4, 0.2])
        response = stub.Predict(request)
        print("Predicted class:", response.label)

if __name__ == "__main__":
    run()

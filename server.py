import grpc
from concurrent import futures
import joblib
import mlservice_pb2
import mlservice_pb2_grpc

# Load trained model
model = joblib.load("iris_model.pkl")

class ModelService(mlservice_pb2_grpc.ModelServiceServicer):
    def Predict(self, request, context):
        features = [request.features]  # gRPC sends as list
        prediction = model.predict(features)[0]
        return mlservice_pb2.PredictReply(label=int(prediction))

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    mlservice_pb2_grpc.add_ModelServiceServicer_to_server(ModelService(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("🚀 ML gRPC server running on port 50051")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()

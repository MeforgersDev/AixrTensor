# model.py
class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                params.extend(layer.parameters())
        return params

    def to(self, device):
        for layer in self.layers:
            if hasattr(layer, 'to'):
                layer.to(device)

    def save_model(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(path):
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)

    def process_important_data(self, important_func, detailed_processing_func=None):
        for param in self.parameters():
            if important_func and important_func(param.data):
                param.to('gpu')
                if detailed_processing_func:
                    detailed_processing_func(param.data)
            else:
                param.to('cpu')

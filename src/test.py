import mnist_loader
import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network.Network([784,30,10])
net.SGD(training_data,10,10,2.0,test_data=test_data)
net.SGD(training_data,10,10,2.0,test_data=test_data)


import mnist_loader
import conv_network

net = conv_network.CNNetwork()
print(net.run(3,2,50))


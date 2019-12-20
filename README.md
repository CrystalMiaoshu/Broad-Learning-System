
# Broad Learning System: An Effective and Efficient Incremental Learning System Without the Need for Deep Architecture

**Chen jun long**






 Bls is a kind of neural network structure which does not depend on depth structure.Compared with the "depth" structure, the "width" structure is very simple because there is no coupling between layers.Similarly, because there is no multi-layer connection and the bls does not need to use gradient descent to update the weights, the computing speed is much better than deep learning.When the accuracy of the network is not up to the requirements, we can increase the "width" of the network to improve the accuracy, and the increase in the width of the calculation and the depth of the network to increase the number of layers, can be said to be very small.The author believes that the bls is suitable for the system with few data features but high demand for real-time prediction.That is to say, this structure does not perform well in large image classification problems like ImageNet, which is why only MNIST and NORB data sets are shown in the original text.


## Concept


The essence of bls is a random vector functional link neural network (RVFLNN).Different from CNN, this network does not change the kernel of feature extractor by reverse transmission, but calculates the weight of each feature node and enhancement node by pseudo-inverse.It's the same as taking 100 kids who don't know math and asking them to solve any problem (like 1+1=?).The answers were any number between 0 and 9, but the children had their own preferences, which meant that their answers were randomly distributed differently.The goal of the network is to predict the answers to the questions from the responses of 100 children.
![](https://github.com/MSZTY/Broad-Learning-System/blob/master/picture/i.jpg)

<img src="picture/i.jpg" width="400" hegiht="213" align=center />




### Contribution

Compared with deep learning, it is a crucial step to propose new ideas on network structure.







## Contacts

Questions about these code should be directed to:

Shu.Miao@hotmail.com

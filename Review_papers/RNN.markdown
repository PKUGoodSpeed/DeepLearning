taidl: review:: Recent Advances in Recurrent Neural Networks

Introduction to RNN:https://arxiv.org/pdf/1801.01078.pdf

- Model Architecture:
![RNN architecture](https://d2mxuefqeaa7sj.cloudfront.net/s_BCBAB4FA40B3313B1FDDA83625EC1F826E4707EEB7803C4900E6C06BE3B33C1D_1517341325557_Screen+Shot+2018-01-30+at+11.40.49+AM.png)

  - Most simple RNN is consist of one input layer, one output layer and a hidden recurrent layer.
  - Input {$$x_t$$}, output {$$y_t$$} and hidden layer {$$h_t$$} are all time series.
  - Recursive formular:
    - $$h_t = f_H(o_t)$$
    - $$o_t = W_{IH}x_t + W_{HH}h_{t-1} + b_{h}$$
    - $$y_t = f_{O}(W_{HO}h_t + b_o)$$


![Most common activations](https://d2mxuefqeaa7sj.cloudfront.net/s_BCBAB4FA40B3313B1FDDA83625EC1F826E4707EEB7803C4900E6C06BE3B33C1D_1517345570374_Screen+Shot+2018-01-30+at+12.52.37+PM.png)




- Activation functions:
  - `'tanh``'` : $$tanh(x) = \frac{e^{2x}-1}{e^{2x}+1}$$
  - `'``sigmoid``'` : $$\sigma(x) = \frac{1}{e^{-x}+1} = \frac{tanh(x/2)+1}{2}$$
  - `'``relu``'` : $$\text{ReLU} (x) = max(x, 0)$$



- Loss Function:
  - $$L(y, y_{\text{pred}})=\sum_t L_t(y_t, y_{\text{pred}, t})$$
  - Euclidean distance/ Hamming distance for regressions
  - Cross entropy for classifications






Training Recurrent Neural Network
This part goes over a bunch of optimization approaches, which are not only specifically used for recurrent networks, including: Gradient Descent, Extended Kalman Filter, Newton’s method, Hessian-Free Optimization, Global Optimization.
If you want to implement an optimizer by yourself, I suggest you to read this part in detail.

![Examples of adding MLP into RNN](https://d2mxuefqeaa7sj.cloudfront.net/s_BCBAB4FA40B3313B1FDDA83625EC1F826E4707EEB7803C4900E6C06BE3B33C1D_1517349402891_Screen+Shot+2018-01-30+at+1.53.37+PM.png)


Recurrent Neural Networks Architecturs

- Deep RNNs with Multi-layer Perceptron: 
  - Deep architectures of neural networks can represent a function exponentially more efficient than shallow architectures.
  - Add intermediate neurons among input, output and hidden layers
  - MLP: multi-layer perception, for which the operations usually includes:
    - addition: $$h + x$$
    - predictor: $$f(Wh + b)$$
  - Multi-hidden layers (stack of hidden layers)


- Bidirectional RNN:
![Illustration of BRNN](https://d2mxuefqeaa7sj.cloudfront.net/s_BCBAB4FA40B3313B1FDDA83625EC1F826E4707EEB7803C4900E6C06BE3B33C1D_1517350987446_Screen+Shot+2018-01-30+at+2.22.56+PM.png)

  - Recursive formula changes a little bit:
    - $$\vec{h}_t = f_H(W_{\vec{IH}}x_t + W_{\vec{HH}}\vec{h}_{t-1}+b_{\vec{h}})$$
    - $$\hat{h}_t = f_H(W_{\hat{IH}}x_t + W_{\hat{HH}}\hat{h}_{t+1}+b_{\hat{h}})$$
    - $$y_t = W_{\vec{HO}}\vec{h}_t + W_{\hat{HO}}\hat{h}_t + b_o$$
  - Training is more complicated, since there are two hidden states one depends on past and the other depends on future.


- RCNN (Recurrent Convolutional Neural Networks):
  - Adding CNN layers before RNN
    - TextCNN: can learn patterns in word embeddings and given the nature of the dataset (e.g. multple misspellings, out of vocabulary words, word phrase correlation)
    - Image capturing
![2D RNN](https://d2mxuefqeaa7sj.cloudfront.net/s_BCBAB4FA40B3313B1FDDA83625EC1F826E4707EEB7803C4900E6C06BE3B33C1D_1517356074069_Screen+Shot+2018-01-30+at+3.47.37+PM.png)



- Multi-Dimensional Recurrent Neural Networks
  - As far as I understand: 


    $$h_{i,j} = W_{HH1}h_{i-1,j} + W_{HH2}h_{i,j-1} + W_{IH}x_{i,j}$$


  for 2D recursion.


![LSTM](https://d2mxuefqeaa7sj.cloudfront.net/s_BCBAB4FA40B3313B1FDDA83625EC1F826E4707EEB7803C4900E6C06BE3B33C1D_1517357102615_Screen+Shot+2018-01-30+at+4.04.51+PM.png)



- Long-Short Term memory (LSTM):
  - Adding gate to determine which information to remember and which information to forget.
![Different types of LSTM](https://d2mxuefqeaa7sj.cloudfront.net/s_BCBAB4FA40B3313B1FDDA83625EC1F826E4707EEB7803C4900E6C06BE3B33C1D_1517357921564_Screen+Shot+2018-01-30+at+4.18.29+PM.png)

![Illustration of GRU](https://d2mxuefqeaa7sj.cloudfront.net/s_BCBAB4FA40B3313B1FDDA83625EC1F826E4707EEB7803C4900E6C06BE3B33C1D_1517358954879_Screen+Shot+2018-01-30+at+4.35.44+PM.png)

-  Gate Recurrent Unit (GRU)
  - Similar to the LSTM unit, the GRU has gating units that modulate the flow of information inside the unit, however, without having separate memory cells.



- Memory network


- Structurally Constrained Recurrent Neural Network


- Unitary Recurrent Neural Networks


- Gated Orthogonal Recurrent Unit


- Hierarchical Subsampling Recurrent Neural Networks

Regularization Recurrent Neural Networks
(The same as other types of networks)

  - $$L_1$$ and $$L_2$$ regularization
  - Dropout
      $$h_t = k\odot h_t$$
  - Activation stabilization
     Adding another term into the loss:
      $$L(y, y_\text{pred}) += \beta\frac{1}{T}\sum_{t=1}^{T}(||h_t||_2 - ||h_{t-1}||_2)^2$$
    ( I thought it should be:
      $$L(y, y_\text{pred}) += \beta\frac{1}{T}\sum_{t=1}^{T}(||h_t - h_{t-1}||_2)^2$$
    , which seems to be more reasonable. )
  - Hidden Activation Preservation (zone out)
    - The zoneout method is a very special case of dropout. It forces some units to keep their activation from the previous time step (i.e. $$h_t = h_{t-1}$$). This approach injects stochasticity (noise) into the network, which makes the network more robust to changes in the hidden state and help the network to avoid overfitting. 
         $$h_t = k\odot h_t + (1-k)\odot 1$$
      (According to the description, it should be $$h_t = k\odot h_t + (1-k)\odot h_{t-1}$$)


Applications:

- Text
- Speech and Audio
- Image
- Video

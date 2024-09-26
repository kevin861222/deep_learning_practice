# impliment of two layers network without pytorch
# build backpropagation
# Version : Python 3.9.17

from argparse import ArgumentParser , ArgumentTypeError
import numpy as np
import matplotlib.pyplot as plt

# training data
def generate_linear_train( n=100 ,seed=2):
    np.random.seed(seed) # 固定seed只是為了方便debug
    points = np.random.uniform(0 , 1 , (n , 2))
    inputs = []
    labels = []
    for point in points :
        inputs.append([point[0],point[1]])
        if point[0] > point[1] :
            labels.append(0)
        else :
            labels.append(1)
    return np.array(inputs) , np.array(labels).reshape(n,1)

# testing data
def generate_linear_test( n=100 ):
    np.random.seed(1) # 生成和Training不一樣的資料
    points = np.random.uniform(0 , 1 , (n , 2))
    inputs = []
    labels = []
    for point in points :
        inputs.append([point[0],point[1]])
        if point[0] > point[1] :
            labels.append(0)
        else :
            labels.append(1)
    return np.array(inputs) , np.array(labels).reshape(n,1)

def generate_XOR_easy() :
    inputs = []
    labels = []
    for i in range(11) :
        inputs.append([0.1*i , 0.1*i])
        labels.append(0)
        if (0.1*i==0.5):
            continue
        inputs.append([0.1*i , 1-(0.1*i)])
        labels.append(1)
    return np.array(inputs) , np.array(labels).reshape(21,1)

def show_result(x ,y ,pred_y,idx=0) :
    plt.subplot(1,2,1)
    plt.title('Ground Truth' , fontsize = 18)
    # filename = 'fig'+str(idx)+'.png'
    for i in range(x.shape[0]):
        if y[i]==0 :
            plt.plot(x[i][0],x[i][1],'ro')
        else :
            plt.plot(x[i][0],x[i][1],'bo')

    plt.subplot(1,2,2)
    plt.title('Predict Result' , fontsize = 18)
    for i in range(x.shape[0]):
        if pred_y[i]==0 :
            plt.plot(x[i][0],x[i][1],'ro')
        else :
            plt.plot(x[i][0],x[i][1],'bo')
    plt.show()
    # plt.savefig(filename)

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def derivative_sigmoid(x):
    return np.multiply(x , 1.0-x )

def relu(x):
    return np.maximum(0.0, x)

def derivative_relu(x):
    return np.heaviside(x, 0.0)

def tanh(x):
    return np.tanh(x)

def derivative_tanh(x):
    return 1.0 - x ** 2

class two_layer_network ():
    def __init__(self,neurals_input,neurals_hl_1,neurals_hl_2,neurals_output,learning_rate,optimize_method,weight_initialization ,convolutional_layer) -> None:
        Din_w1 = neurals_input
        Dout_w1 = neurals_hl_1
        Din_w2 = Dout_w1
        Dout_w2 = neurals_hl_2
        Din_w3 = Dout_w2
        Dout_w3 = neurals_output
        np.random.seed(2) # easier to debug
        if (weight_initialization=='normal'):
            self.w1 = np.random.normal(0 , 1 ,(Din_w1,Dout_w1))
            self.w2 = np.random.normal(0 , 1 ,(Din_w2,Dout_w2))
            self.w3 = np.random.normal(0 , 1 ,(Din_w3,Dout_w3))

        elif (weight_initialization=='randn'):
            self.w1 = np.random.randn(Din_w1,Dout_w1) *0.01 # avg = 0 std_deviation = 0.01
            self.w2 = np.random.randn(Din_w2,Dout_w2) *0.01 # avg = 0 std_deviation = 0.01
            self.w3 = np.random.randn(Din_w3,Dout_w3) *0.01 # avg = 0 std_deviation = 0.01
        else :
            raise ArgumentTypeError('weight_initialization : normal or randn')

        self.learning_rate = learning_rate
        if (optimize_method == 'gdm'):
            self.w1_m = np.zeros((Din_w1,Dout_w1))
            self.w2_m = np.zeros((Din_w2,Dout_w2))
            self.w3_m = np.zeros((Din_w3,Dout_w3))
        self.losslist = []

        if (convolutional_layer):
            # kernel_shape = (1,2)
            self.kernel1 = np.random.randn(*(1,2))
            self.kernel2 = np.random.randn(*(1,1))

    def forwardpass(self,input_data,active_funtion,convolutional_layer):
        # print('forward')
        # print(self.w3)
        if (convolutional_layer):
            if (active_funtion == 'sigmoid'):
                conv_out1 = convolution(input_data, self.kernel1)
                act_out1 = sigmoid(conv_out1)
                conv_out2 = convolution(act_out1, self.kernel2)
                act_out2 = sigmoid(conv_out2)
                self.pred_y = act_out2
            
            elif (active_funtion == 'relu'):
                conv_out1 = convolution(input_data, self.kernel1)
                act_out1 = relu(conv_out1)
                conv_out2 = convolution(act_out1, self.kernel2)
                act_out2 = relu(conv_out2)
                self.pred_y = act_out2
            
            elif (active_funtion == 'tanh'):
                conv_out1 = convolution(input_data, self.kernel1)
                act_out1 = tanh(conv_out1)
                conv_out2 = convolution(act_out1, self.kernel2)
                act_out2 = tanh(conv_out2)
                self.pred_y = act_out2
            
            elif (active_funtion == 'none'):
                conv_out1 = convolution(input_data, self.kernel1)
                act_out1 = conv_out1
                conv_out2 = convolution(act_out1, self.kernel2)
                act_out2 = conv_out2
                self.pred_y = act_out2
            
            else :
                raise ArgumentTypeError('active function : sigmoid , relu , tanh or None')
            
        else :
            if (active_funtion == 'sigmoid'):
                self.z1 = np.dot( input_data , self.w1 )
                self.a1 = sigmoid(self.z1)
                self.z2 = np.dot( self.a1 , self.w2 )
                self.a2 = sigmoid(self.z2)
                self.z3 = np.dot( self.a2 , self.w3 )
                self.pred_y = sigmoid(self.z3)

            elif (active_funtion == 'relu'):
                self.z1 = np.dot( input_data , self.w1 )
                self.a1 = relu(self.z1)
                self.z2 = np.dot( self.a1 , self.w2 )
                self.a2 = relu(self.z2)
                self.z3 = np.dot( self.a2 , self.w3 )
                self.pred_y = relu(self.z3)

            elif (active_funtion == 'tanh'):
                self.z1 = np.dot( input_data , self.w1 )
                self.a1 = tanh(self.z1)
                self.z2 = np.dot( self.a1 , self.w2 )
                self.a2 = tanh(self.z2)
                self.z3 = np.dot( self.a2 , self.w3 )
                self.pred_y = tanh(self.z3)

            elif (active_funtion == 'none'):
                self.z1 = np.dot( input_data , self.w1 )
                self.a1 = (self.z1)
                self.z2 = np.dot( self.a1 , self.w2 )
                self.a2 = (self.z2)
                self.z3 = np.dot( self.a2 , self.w3 )
                self.pred_y = (self.z3)

            else :
                raise ArgumentTypeError('active function : sigmoid , relu , tanh or None')
        # ------------shape------------SHOW UP : command+K then commend+U
        # print('input',input_data.shape)         # input (100, 2)
        # print('w1',self.w1.shape)               # w1 (2, 4)
        # print('z1',self.z1.shape)               # z1 (100, 4)        
        # print('a1',self.a1.shape)               # a1 (100, 4)
        # print('w2',self.w2.shape)               # w2 (4, 4)
        # print('z2',self.z2.shape)               # z2 (100, 4)
        # print('a2',self.a2.shape)               # a2 (100, 4)
        # print('z3',self.z3.shape)               # z3 (100, 1)
        # print('w3',self.w3.shape)               # w3 (4, 1)
        # print('pred_y',self.pred_y.shape)       # pred_y (100, 1)
        # ------------shape------------Annotations : command+K then commend+C
        return None
    
    def MSE_loss(self,ground_truth):
        N = len(self.pred_y)
        self.loss = np.sum(np.power((ground_truth - self.pred_y),2)) / (N)
        self.losslist.append(self.loss)
        return self.loss
        
    def backward_pass(self,ground_truth,active_funtion):
        if (active_funtion == 'sigmoid'):
            self.c_y = 2*(self.pred_y - ground_truth) / (self.pred_y.shape[0])
            self.c_z3 = derivative_sigmoid(self.pred_y)*self.c_y 
            self.c_z2 = derivative_sigmoid(self.a2)*np.dot(self.c_z3,self.w3.T)
            self.c_z1 = derivative_sigmoid(self.a1)*np.dot(self.c_z2,self.w2.T)
        
        elif (active_funtion == 'relu'):
            self.c_y = 2*(self.pred_y - ground_truth) / (self.pred_y.shape[0])
            self.c_z3 = derivative_relu(self.pred_y)*self.c_y 
            self.c_z2 = derivative_relu(self.a2)*np.dot(self.c_z3,self.w3.T)
            self.c_z1 = derivative_relu(self.a1)*np.dot(self.c_z2,self.w2.T)

        elif (active_funtion == 'tanh'):
            self.c_y = 2*(self.pred_y - ground_truth) / (self.pred_y.shape[0])
            self.c_z3 = derivative_tanh(self.pred_y)*self.c_y 
            self.c_z2 = derivative_tanh(self.a2)*np.dot(self.c_z3,self.w3.T)
            self.c_z1 = derivative_tanh(self.a1)*np.dot(self.c_z2,self.w2.T)

        elif (active_funtion == 'none'):
            self.c_y = 2*(self.pred_y - ground_truth) / (self.pred_y.shape[0])
            self.c_z3 = self.c_y 
            self.c_z2 = np.dot(self.c_z3,self.w3.T)
            self.c_z1 = np.dot(self.c_z2,self.w2.T)

        else :
            raise ArgumentTypeError('active function : sigmoid , relu or tanh')
        # ------------shape------------SHOW UP : command+K then commend+U
        # print('c_z1',self.c_z1.shape)             # c_z1 (100, 4)
        # print('c_z2',self.c_z2.shape)             # c_z2 (100, 4)
        # print('c_z3',self.c_z3.shape)             # c_z3 (100, 1)
        # ------------shape------------Annotations : command+K then commend+C
        return None

    def gradient(self,input_data):
        # self.c_w1 = np.dot(input_data.T,self.c_z1)
        self.c_w1 = gradient_cal(input_data , self.c_z1 , self.w1.shape)
        # self.c_w2 = np.dot(self.a1.T , self.c_z2 )
        self.c_w2 = gradient_cal(self.a1 , self.c_z2 ,self.w2.shape)
        # self.c_w3 = np.dot(self.a2.T , self.c_z3 )
        self.c_w3 = gradient_cal(self.a2 , self.c_z3 ,self.w3.shape)
        # ------------shape------------SHOW UP : command+K then commend+U
        # print('c_w1',self.c_w1.shape)             # c_w1 (2, 4)
        # print('c_w2',self.c_w2.shape)             # c_w2 (4, 4)
        # print('c_w3',self.c_w3.shape)             # c_w3 (4, 1)
        # ------------shape------------Annotations : command+K then commend+C
        return None    

    def update(self,optimize_method):
        if (optimize_method == 'gd'):
            self.w1 = self.w1 - (self.learning_rate *self.c_w1)
            self.w1 = np.clip(self.w1,-1024. , 1023.)

            self.w2 = self.w2 - (self.learning_rate *self.c_w2)
            self.w2 = np.clip(self.w2,-1024. , 1023.)

            self.w3 = self.w3 - (self.learning_rate *self.c_w3)
            self.w3 = np.clip(self.w3,-1024. , 1023.)
        elif (optimize_method == 'gdm'):
            self.w1 = self.w1 - (self.learning_rate *self.c_w1) - 0.8*self.w1_m
            self.w1 = np.clip(self.w1,-1024. , 1023.)
            self.w1_m = (self.learning_rate *self.c_w1)

            self.w2 = self.w2 - (self.learning_rate *self.c_w2) - 0.8*self.w2_m
            self.w2 = np.clip(self.w2,-1024. , 1023.)
            self.w2_m = (self.learning_rate *self.c_w2)

            self.w3 = self.w3 - (self.learning_rate *self.c_w3) - 0.8*self.w3_m
            self.w3 = np.clip(self.w3,-1024. , 1023.)
            self.w3_m = (self.learning_rate *self.c_w3)
            
        else :
            raise ArgumentTypeError('optimize_method : gd or gdm')
        # ------------shape------------SHOW UP : command+K then commend+U
        # print('w1',self.w1.shape)
        # print('w2',self.w2.shape)
        # print('w3',self.w3.shape)
        # ------------shape------------Annotations : command+K then commend+C
        return None
    
    def backward_conv(self, input_data, grad_output, learning_rate):
        input_height, input_width = input_data.shape
        grad_output_height, grad_output_width = grad_output.shape
        kernel1_height, kernel1_width = self.kernel1.shape
        kernel2_height, kernel2_width = self.kernel2.shape

        grad_kernel1 = np.zeros_like(self.kernel1)
        grad_kernel2 = np.zeros_like(self.kernel2)

        # 反向傳播計算梯度
        for i in range(grad_output_height):
            for j in range(grad_output_width):
                grad_kernel2 += relu(convolution(input_data=input_data[i:i+kernel2_height, j:j+kernel2_width],kernel= grad_output[i, j]))
                grad_output_relu1 = convolution(input_data=grad_output[i, j].reshape(1, 1), kernel= np.rot90(self.kernel2, 2))
                grad_output_relu1 = grad_output_relu1 * (convolution(input_data[i:i+kernel2_height, j:j+kernel2_width], np.rot90(self.kernel2, 2)) > 0)
                grad_kernel1 += input_data[i:i+kernel1_height, j:j+kernel1_width] * grad_output_relu1

        # 使用隨機梯度下降法更新參數
        self.kernel1 -= learning_rate * grad_kernel1
        self.kernel2 -= learning_rate * grad_kernel2

        # 計算輸入的梯度用於上一層的反向傳播
        grad_input = np.zeros_like(input_data)
        for i in range(grad_output_height):
            for j in range(grad_output_width):
                grad_input[i:i+kernel1_height, j:j+kernel1_width] += np.rot90(self.kernel1, 2) * grad_output[i, j]

        return grad_input

def gradient_cal(forward_gradient , backward_gradient ,refshape):

    if (len(forward_gradient)==len(backward_gradient)):
        result = np.zeros(refshape)
        for i in range (len(forward_gradient)):
            B_array = np.array([backward_gradient[i]])
            F_array = np.array([forward_gradient[i]])
            result = result + np.dot(F_array.T , B_array)
    else :
        raise IndexError('len(forward_gradient)!=len(backward_gradient)')
    result = np.clip(result,-1024. , 1023.)
    return result

def learning_curve(loss,data_source):
    plt.plot(loss)
    plt.title(f'{data_source}')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid(True)
    plt.show()
    return None

def convolution(input_data, kernel):
    input_height, input_width = input_data.shape
    kernel_height, kernel_width = kernel.shape
    output_height = input_height - kernel_height + 1
    output_width = input_width - kernel_width + 1

    output = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            output[i, j] = np.sum(input_data[i:i+kernel_height, j:j+kernel_width] * kernel)

    return output

def parser():
    parser = ArgumentParser(description="NTCU-DLP-Lab01\tBackpropagation")
    parser.add_argument('-H1', '--neurals_hl_1', default=4, type=int, help='Number of neurals in hidden layer1')
    parser.add_argument('-H2', '--neurals_hl_2', default=4, type=int, help='Number of neurals in hidden layer2')
    parser.add_argument('-DS', '--data_source', default='Linear', type=str, help='The data source for training and testing\n\t: Linear or XOR')
    parser.add_argument('-LR', '--learning_rate', default=0.1, type=float, help='Learning rate of the neural network')
    parser.add_argument('-AF', '--active_funtion', default='sigmoid', type=str, help='The active finction of neurals\n\t: sigmoid\n\trelu\n\ttanh\n\tnone')
    parser.add_argument('-OM', '--optimize_method', default='gd', type=str, help='The optimize method of neural network\n\t:\tgd : gradient descent\n\tgdm : gradient descent with momentum')
    parser.add_argument('-WI', '--weight_initialization', default='normal', type=str, help='The method to initialize the weights\n\tnormal\n\trandn : avg = 0 std_deviation = 0.01')
    parser.add_argument('-DR', '--data_reproduction', default=False, type=bool, help='whether reproduce new linear data after 500 epoch')
    parser.add_argument('-CL', '--convolutional_layer', default=False, type=bool, help='whether to add convolutional layers')
    return parser.parse_args()

def main():
    fig_idx = 0
    epoch = int(20000)
    arguments = parser()
    # training settings
    learning_rate = arguments.learning_rate
    active_funtion = arguments.active_funtion
    optimize_method = arguments.optimize_method
    weight_initialization = arguments.weight_initialization
    data_reproduction = arguments.data_reproduction
    convolutional_layer = arguments.convolutional_layer
    # network parameters
    neurals_input = 2
    neurals_hl_1 = arguments.neurals_hl_1
    neurals_hl_2 = arguments.neurals_hl_2
    neurals_output = 1
    data_source = arguments.data_source

    if data_source=='Linear':
        input_data , label = generate_linear_train()
    elif data_source=='XOR':
        input_data , label = generate_XOR_easy()
    else :
        raise ArgumentTypeError('data_source : Linear or XOR')
    network = two_layer_network(neurals_input=neurals_input , neurals_hl_1=neurals_hl_1 , neurals_hl_2=neurals_hl_2 , neurals_output=neurals_output , learning_rate=learning_rate , optimize_method=optimize_method , weight_initialization=weight_initialization , convolutional_layer=convolutional_layer)
    if (data_source == 'Linear'):
        f = open('Linear_output.txt', 'w')
    elif(data_source == 'XOR'):
        f = open('XOR_output.txt', 'w')
    f.write(f'| training setting |\n\tdata : {data_source}\n\tactive finction : {active_funtion}\n\toptimize method : {optimize_method}\n\tweight initialization : {weight_initialization}\n\tepoch : {epoch}\n\tdata_reproduction : {data_reproduction}\n\tconvolutional_layer : {convolutional_layer}\n\n')
    f.write(f'| network parameters |\n\th1 units : {neurals_hl_1}\n\th2 units : {neurals_hl_2}\n\n')
    # train
    if (convolutional_layer):
        for j in range (int(epoch / 500)):
            for i in range (500):
                network.forwardpass(input_data=input_data,active_funtion=active_funtion,convolutional_layer=convolutional_layer)
                grad_output = network.pred_y - label
                loss = network.MSE_loss(ground_truth=label)
                network.backward_conv(grad_output=grad_output , input_data=input_data , learning_rate=learning_rate)
                if (np.isnan(loss)):
                    print(i,'nan')
                    break
                else :
                    pass

            print((j+1)*5000,'epoch : loss =',loss) #),'network.loss =',network.loss)
            if ((data_source=='Linear')&(data_reproduction)):
                input_data , label = generate_linear_train(seed=j+3)
    else :
        for j in range (int(epoch / 500)):
            for i in range (500):
                network.forwardpass(input_data=input_data,active_funtion=active_funtion,convolutional_layer=convolutional_layer)
                loss = network.MSE_loss(ground_truth=label)
                network.backward_pass(ground_truth=label,active_funtion=active_funtion)
                network.gradient(input_data=input_data)
                network.update(optimize_method=optimize_method)
                if (np.isnan(loss)):
                    print(i,'nan')
                    break
                else :
                    pass

            print((j+1)*5000,'epoch : loss =',loss) #),'network.loss =',network.loss)
            if ((data_source=='Linear')&(data_reproduction)):
                input_data , label = generate_linear_train(seed=j+3)
            
    f.write(f'| result |\n\ttrainning loss = {loss}\n\n')
    learning_curve(network.losslist,data_source=data_source)
    # test
    if (data_source=='Linear'):
        input_data , label = generate_linear_test()
    elif (data_source=='XOR'):
        input_data , label = generate_XOR_easy()
    else :
        raise ArgumentTypeError('data_source : Linear or XOR')
    network.forwardpass(input_data=input_data,active_funtion=active_funtion)
    loss = network.MSE_loss(ground_truth=label)
    for i in range (network.pred_y.shape[0]):
        print('Iter ',i,'\tground truth =',label[i][0],'\tprediction =',network.pred_y[i][0])

    network.pred_y[network.pred_y>0.5] = 1
    network.pred_y[network.pred_y <=0.5] = 0
    acc = 0

    for i in (network.pred_y == label):
        if (i) :
            acc = acc+1
    acc = acc / network.pred_y.shape[0]
    print('acc =',acc)
    f.write(f'\ttesting loss = {loss}\n\taccuracy = {acc}\n\n')
    show_result(x=input_data,y=label,pred_y=network.pred_y)
    f.close()

if __name__ == '__main__':
    main()


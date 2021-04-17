import os
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.quantization
import copy 
import os
import zipfile
import tempfile
import shutil

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)

class LSTM(nn.Module):
    def __init__(self, num_classes, hidden_size1=32, hidden_size2=32, num_layers=2):
        super(LSTM, self).__init__()
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.num_layers = num_layers
        self.embedding= nn.Embedding(num_classes, 32).to(device)
        self.lstm = nn.LSTM(32, hidden_size1, num_layers, batch_first=True).to(device)
        self.fc1 = nn.Linear(hidden_size1, hidden_size2).to(device)
        self.fc2 = nn.Linear(hidden_size2, num_classes).to(device)
        

    def forward(self, x):
        # initialize
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size1).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size1).to(device)
        
        out=self.embedding(x[:,:,0].long())
        out, (h_n, h_c) = self.lstm(out, (h0, c0))
        out =self.fc1(out[:, -1, :])
        out =self.fc2(nn.ReLU()(out))

        return out

def train_model(config):
    result=[]
    
    percentage_ls=config.percentage_ls

    batch_size=128
    time_steps= 64

    def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
            nrows = ((a.size - L) // S) + 1
            n = a.strides[0]
            return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n), writeable=False)

    # load the preprocessed data
    series = np.load(config.file_path)
    series = series.reshape(-1, 1)

    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit(series)
    series = series.reshape(-1)

    data = strided_app(series, time_steps+1, 1)
    l = int(len(data)/batch_size) * batch_size

    data = data[:l] 
    X = data[:, :-1]
    Y = data[:, -1:].reshape([-1,])
    Y_hot = onehot_encoder.transform(data[:, -1:])

    # Hyper Parameters
    input_size = 1   
    num_epochs_retrain= config.num_epochs_retrain
    num_epochs= config.num_epochs
    hidden_size1 = config.hidden_size1
    hidden_size2 = config.hidden_size2
    num_layers = config.num_layers
    num_classes = Y_hot.shape[1]
    lr = config.lr

    model = LSTM(num_classes,hidden_size1, hidden_size2, num_layers)

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)

    # load training data
    train_data = TensorDataset(torch.Tensor(X),torch.Tensor(Y).long())
    train_loader = DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True) 
    test_data = TensorDataset(torch.Tensor(X),torch.Tensor(Y).long())
    test_loader = DataLoader(dataset=test_data,batch_size=batch_size,shuffle=True) 

    # train
    total_step = len(train_loader)
    model_ls=[]
    accuracy=[]
    for epoch in range(num_epochs):
        for i, (x, y) in enumerate(train_loader):
            x = x.reshape(-1, time_steps, input_size).to(device)
            y = y.to(device)

            # forward pass
            outputs = model(x)
            loss = criterion(outputs, y)

            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 1000 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
                
                with torch.no_grad():
                    correct = 0
                    total = 0
                    times=0
                    for x, y in test_loader:
                        x = x.reshape(-1, time_steps, input_size).to(device)
                        y = y.to(device)
                        outputs = model(x)
                        prob=F.softmax(outputs)
                        _, predicted = torch.max(prob, 1)
                        total += y.size(0)
                        correct += (predicted == y).sum().item()
                        times=times+1
                        if times > 100:
                            break

                    print('Test Accuracy of the model on the 10000 test x: {} %'.format(100 * correct / total))
                model_ls.append(model)
                accuracy.append(100 * correct / total)

    model=model_ls[np.argmax(accuracy)]

    # test accuary
    test_data = TensorDataset(torch.Tensor(X),torch.Tensor(Y).long())
    test_loader = DataLoader(dataset=test_data,batch_size=batch_size,shuffle=True) 
    with torch.no_grad():
        correct = 0
        total = 0
        times=0
        for x, y in test_loader:
            x = x.reshape(-1, time_steps, input_size).to(device)
            y = y.to(device)
            outputs = model(x)
            prob=F.softmax(outputs)
            _, predicted = torch.max(prob, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            times=times+1
            if times > 100:
                break

        print('Test Accuracy of the model on the 10000 test x: {} %'.format(100 * correct / total))

    # # quantize the unpruned model
    # quantized_model = torch.quantization.quantize_dynamic(
    #     model.to('cpu'), {nn.Embedding,nn.LSTM, nn.Linear}, dtype=torch.qint8
    # )

    def pruning(model0,percentage,method):
        # copy a model0 for pruning
        # model0=copy.deepcopy(model.to(device))
        if method=="unstructured":
            for name, module in model0.named_modules():
                if isinstance(module, torch.nn.Embedding):
                    prune.l1_unstructured(module, name='weight', amount=percentage)
                # prune lstm layers
                elif isinstance(module, torch.nn.LSTM):
                    prune.l1_unstructured(module, name='weight_hh_l0', amount=percentage)
                    prune.l1_unstructured(module, name='weight_ih_l0', amount=percentage)
                    prune.l1_unstructured(module, name='weight_hh_l1', amount=percentage)
                    prune.l1_unstructured(module, name='weight_ih_l1', amount=percentage)
                # prune  linear layers
                elif isinstance(module, torch.nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=percentage)
        elif method=="structured":
            for name, module in model0.named_modules():
                if isinstance(module, torch.nn.Embedding):
                    prune.ln_structured(module, name='weight', amount=percentage,n=1,dim=0)
                # prune lstm layers
                elif isinstance(module, torch.nn.LSTM):
                    prune.ln_structured(module, name='weight_hh_l0', amount=percentage,n=1,dim=0)
                    prune.ln_structured(module, name='weight_ih_l0', amount=percentage,n=1,dim=0)
                    prune.ln_structured(module, name='weight_hh_l1', amount=percentage,n=1,dim=0)
                    prune.ln_structured(module, name='weight_ih_l1', amount=percentage,n=1,dim=0)
                # prune  linear layers
                elif isinstance(module, torch.nn.Linear):
                    prune.ln_structured(module, name='weight', amount=percentage,n=1,dim=0)            
        for name, module in model0.named_modules():
            if isinstance(module, torch.nn.Embedding):
                prune.remove(module, 'weight')
            # prune  lstm layers
            elif isinstance(module, torch.nn.LSTM):
                prune.remove(module, 'weight_hh_l0')
                prune.remove(module, 'weight_ih_l0')
                prune.remove(module, 'weight_hh_l1')
                prune.remove(module, 'weight_ih_l1')
            # prune  linear layers
            elif isinstance(module, torch.nn.Linear):
                prune.remove(module, 'weight')

        test_data = TensorDataset(torch.Tensor(X),torch.Tensor(Y).long())
        test_loader = DataLoader(dataset=test_data,batch_size=batch_size,shuffle=True) 
        # model_prune=copy.deepcopy(model0.to(device))
        model_prune=model0
        optimizer = torch.optim.Adam(model_prune.parameters(), lr)
        model_ls=[]
        accuracy=[]
        for epoch in range(num_epochs_retrain):
            for i, (x, y) in enumerate(train_loader):
                x = x.reshape(-1, time_steps, input_size).to(device)
                y = y.to(device)

                # forward pass
                outputs = model_prune(x)
                loss = criterion(outputs, y)

                # backward and optimize
                optimizer.zero_grad()
                loss.backward()

                for name, param in model_prune.named_parameters():
                    if "weight" in name:
                        param_data = param.data.cpu().numpy()
                        param_grad = param.grad.data.cpu().numpy()
                        param_grad = np.where(param_data < 0.00001, 0, param_grad)
                        param.grad.data = torch.from_numpy(param_grad).to(device)

                optimizer.step()

                if i % 1000 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                          .format(epoch+1, num_epochs_retrain, i+1, total_step, loss.item()))
                    
                    with torch.no_grad():
                        correct = 0
                        total = 0
                        times=0
                        for x, y in test_loader:
                            x = x.reshape(-1, time_steps, input_size).to(device)
                            y = y.to(device)
                            outputs = model_prune(x)
                            prob=F.softmax(outputs)
                            _, predicted = torch.max(prob, 1)
                            total += y.size(0)
                            correct += (predicted == y).sum().item()
                            times=times+1
                            if times > 100:
                                break

                        print('Test Accuracy of the model on the 10000 test x: {} %'.format(100 * correct / total))   
                    model_ls.append(model_prune)
                    accuracy.append(100 * correct / total)

        model_prune=model_ls[np.argmax(accuracy)]

        # test accuary
        test_data = TensorDataset(torch.Tensor(X),torch.Tensor(Y).long())
        test_loader = DataLoader(dataset=test_data,batch_size=batch_size,shuffle=True) 
        with torch.no_grad():
            correct = 0
            total = 0
            times=0
            for x, y in test_loader:
                x = x.reshape(-1, time_steps, input_size).to(device)
                y = y.to(device)
                outputs = model_prune(x)
                prob=F.softmax(outputs)
                _, predicted = torch.max(prob, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
                times=times+1
                if times > 100:
                    break

        print('Test Accuracy of the model on the 10000 test x: {} %'.format(100 * correct / total))

        # quantize the pruned model
        quantized_model_prune = torch.quantization.quantize_dynamic(
            model_prune.to('cpu'), {nn.Embedding,nn.LSTM, nn.Linear}, dtype=torch.qint8
        )

        return model_prune,quantized_model_prune

    # model_base,quantized_model_base=pruning(copy.deepcopy(model).to(device),0,"unstructured")
    # model_prune_20,quantized_model_prune_20=pruning(copy.deepcopy(model).to(device),0.2,"unstructured")
    # model_prune_50,quantized_model_prune_50=pruning(copy.deepcopy(model).to(device),0.5,"unstructured")
    # model_prune_80,quantized_model_prune_80=pruning(copy.deepcopy(model).to(device),0.8,"unstructured")

    # model_prune_20_struct,quantized_model_prune_20_struct=pruning(copy.deepcopy(model).to(device),0.2,"structured")
    # model_prune_50_struct,quantized_model_prune_50_struct=pruning(copy.deepcopy(model).to(device),0.5,"structured")
    # model_prune_80_struct,quantized_model_prune_80_struct=pruning(copy.deepcopy(model).to(device),0.8,"structured")

    def cal_size(model_test):
        param_dict=[model_test.embedding.weight,model_test.lstm.bias_ih_l0,model_test.lstm.bias_hh_l0,
                    model_test.lstm.weight_ih_l1,model_test.lstm.weight_hh_l1,model_test.lstm.bias_ih_l1,
                    model_test.lstm.bias_hh_l1,model_test.lstm.weight_hh_l0,model_test.lstm.weight_ih_l0,
                    model_test.fc1.bias,model_test.fc1.weight,model_test.fc2.bias,model_test.fc2.weight]
        size_origin=0
        size_quantized=0
        for weight in param_dict:
            model_array=weight.cpu().detach().numpy().reshape([-1,])
            size_origin=size_origin+len(model_array[model_array!=0])*32+len(model_array[model_array==0])
            size_quantized=size_quantized+len(model_array[model_array!=0])*8+len(model_array[model_array==0])
        return size_origin/1024,size_quantized/1024

    # # print("baseline: "+str(cal_size(model)))
    # print("baseline: "+str(cal_size(model_base)))
    # result.append("baseline: "+str(cal_size(model_base)))

    # print("prune 20%: "+str(cal_size(model_prune_20)))
    # result.append("prune 20%: "+str(cal_size(model_prune_20)))

    # print("prune 50%: "+str(cal_size(model_prune_50)))
    # result.append("prune 50%: "+str(cal_size(model_prune_50)))

    # print("prune 80%: "+str(cal_size(model_prune_80)))
    # result.append("prune 80%: "+str(cal_size(model_prune_80)))

    # print("prune_struct 20%: "+str(cal_size(model_prune_20_struct)))
    # result.append("prune_struct 20%: "+str(cal_size(model_prune_20_struct)))

    # print("prune_struct 50%: "+str(cal_size(model_prune_50_struct)))
    # result.append("prune_struct 50%: "+str(cal_size(model_prune_50_struct)))

    # print("prune_struct 80%: "+str(cal_size(model_prune_80_struct)))
    # result.append("prune_struct 80%: "+str(cal_size(model_prune_80_struct)))

    def save_model(model,file_name):
        torch.save(model.state_dict(), "../data/trained_models/"+config.data_name+"/temptory/"+file_name)

        # define a gzip function
        def get_gzipped_model_size(file,path):
          # Returns size of gzipped model, in bytes.
          _, zipped_file = tempfile.mkstemp('.zip')
          with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
            f.write(file)
          shutil.copy(zipped_file, path)
          return os.path.getsize(zipped_file)

        os.chdir("../data/trained_models/"+config.data_name+"/temptory")

        save_path="../"+file_name+".zip"

        string="Size of gzipped "+ file_name + " model: %.2f KB" % (get_gzipped_model_size(file_name,save_path)/1024)
        print(string)

        os.chdir("../../../../src")

        return string

    model_base,quantized_model_base=pruning(copy.deepcopy(model).to(device),0,"unstructured")
    print("baseline: "+str(cal_size(model_base)))
    result0=[]
    result0.append("baseline: "+str(cal_size(model_base)))
    result0.append("lstm_baseline")
    result0.append(save_model(model_base,"lstm_baseline"+config.append))
    result0.append("lstm_quantized")
    result0.append(save_model(quantized_model_base,"lstm_quantized"+config.append))
    result.append(result0)

    for percentage in percentage_ls:
        model_temp,quantized_model_temp=pruning(copy.deepcopy(model).to(device),percentage,"unstructured")
        print("prune "+str(int(percentage*100))+"%: "+str(cal_size(model_temp)))
        result0=[]
        result0.append("prune "+str(int(percentage*100))+"%: "+str(cal_size(model_temp)))
        result0.append("lstm_pruned_"+str(int(percentage*100)))
        result0.append(save_model(model_temp,"lstm_pruned_"+str(int(percentage*100))+config.append))
        result0.append("lstm_quantized_pruned_"+str(int(percentage*100)))
        result0.append(save_model(quantized_model_temp,"lstm_quantized_pruned_"+str(int(percentage*100))+config.append))   
        result.append(result0) 

    for percentage in percentage_ls:
        model_temp,quantized_model_temp=pruning(copy.deepcopy(model).to(device),percentage,"structured")
        print("prune_struct "+str(int(percentage*100))+"%: "+str(cal_size(model_temp)))
        result0=[]
        result0.append("prune_struct "+str(int(percentage*100))+"%: "+str(cal_size(model_temp)))
        result0.append("lstm_pruned_struct_"+str(int(percentage*100)))
        result0.append(save_model(model_temp,"lstm_pruned_struct_"+str(int(percentage*100))+config.append))
        result0.append("lstm_quantized_pruned_struct_"+str(int(percentage*100)))
        result0.append(save_model(quantized_model_temp,"lstm_quantized_pruned_struct_"+str(int(percentage*100))+config.append))   
        result.append(result0) 

    # result.append(save_model(model_base,"lstm_baseline"+config.append))
    # result.append(save_model(quantized_model_base,"lstm_quantized"+config.append))

    # result.append(save_model(model_prune_20,"lstm_pruned_20"+config.append))
    # result.append(save_model(model_prune_50,"lstm_pruned_50"+config.append))
    # result.append(save_model(model_prune_80,"lstm_pruned_80"+config.append))
    # result.append(save_model(quantized_model_prune_20,"lstm_quantized_pruned_20"+config.append))
    # result.append(save_model(quantized_model_prune_50,"lstm_quantized_pruned_50"+config.append))
    # result.append(save_model(quantized_model_prune_80,"lstm_quantized_pruned_80"+config.append))

    # result.append(save_model(model_prune_20_struct,"lstm_pruned_20_struct"+config.append))
    # result.append(save_model(model_prune_50_struct,"lstm_pruned_50_struct"+config.append))
    # result.append(save_model(model_prune_80_struct,"lstm_pruned_80_struct"+config.append))
    # result.append(save_model(quantized_model_prune_20_struct,"lstm_quantized_pruned_20_struct"+config.append))
    # result.append(save_model(quantized_model_prune_50_struct,"lstm_quantized_pruned_50_struct"+config.append))
    # result.append(save_model(quantized_model_prune_80_struct,"lstm_quantized_pruned_80_struct"+config.append))

    np.save("../result/"+config.data_name+config.append,result)

    return result

def check_model_size(file_name,config):
    os.chdir("../data/trained_models/"+config.data_name)
    print("Size of gzipped "+ file_name +config.append+ " model: %.2f KB" % (os.path.getsize(file_name+config.append+".zip")/1024))    
    os.chdir("../../../src")
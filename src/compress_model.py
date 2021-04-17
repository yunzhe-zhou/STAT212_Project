import os

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import argparse
import contextlib
import arithmeticcoding_fast
import json
from tqdm import tqdm
import struct
import tempfile
import shutil
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import torch.quantization
import zipfile

torch.manual_seed(1)

def compress_model(args,device):
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
    # load the data
    np.random.seed(0)

    series = np.load(args.sequence_npy_file)
    series = series.reshape(-1, 1)

    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit(series)

    args.batch_size=int(len(series)/10000)
    batch_size = args.batch_size
    timesteps = 64

    with open(args.params_file, 'r') as f:
            params = json.load(f)

    params['len_series'] = len(series)
    params['bs'] = batch_size
    params['timesteps'] = timesteps

    with open(args.output_file_prefix+'.params','w') as f:
            json.dump(params, f, indent=4)

    alphabet_size = len(params['id2char_dict'])

    def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
        nrows = ((a.size - L) // S) + 1
        n = a.strides[0]
        return np.lib.stride_tricks.as_strided(
            a, shape=(nrows, L), strides=(S * n, n), writeable=False)

    series = series.reshape(-1)
    data = strided_app(series, timesteps+1, 1)

    X = data[:, :-1]
    Y_original = data[:, -1:]
    Y = onehot_encoder.transform(Y_original)

    l = int(len(series)/batch_size)*batch_size

    # Hyper Parameters       
    num_epochs=args.num_epochs      
    hidden_size1 = args.hidden_size1
    hidden_size2 = args.hidden_size2
    num_layers = args.num_layers
    num_classes = alphabet_size
    lr = 0.001 


    if args.model_name=="LSTM":
        model = LSTM(num_classes,hidden_size1, hidden_size2, num_layers)
    elif args.model_name=="GRU":
        model = GRU(num_classes,hidden_size1, hidden_size2, num_layers)
    elif args.model_name=="biLSTM":
        model = biLSTM(num_classes,hidden_size1, hidden_size2, num_layers)
    elif args.model_name=="biGRU": 
        model = biGRU(num_classes,hidden_size1, hidden_size2, num_layers)

    # unzip the compressed models
    zip_path="../data/trained_models/"+args.data_name+"/"+args.file_path+args.append+ ".zip"
    save_path="../data/trained_models/"+args.data_name
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(save_path)

    model=model.to('cpu')
    if args.quantization=="yes":
        # load the quantized model weights
        model = torch.quantization.quantize_dynamic(
            model, {nn.LSTM, nn.Linear}, dtype=torch.qint8
        )
        model.load_state_dict(torch.load(args.model_weights_file))
    else:
        #load the model weights
        model.load_state_dict(torch.load(args.model_weights_file))
    model=model.to(device)
        
    def predict_lstm(X, y, y_original, timesteps, bs, alphabet_size, model_name, final_step=False):       
            if not final_step:
                    num_iters = int((len(X)+timesteps)/bs)
                    ind = np.array(range(bs))*num_iters
                    
                    # open compressed files and compress first few characters using
                    # uniform distribution
                    f = [open(args.temp_file_prefix+'.'+str(i),'wb') for i in range(bs)]
                    bitout = [arithmeticcoding_fast.BitOutputStream(f[i]) for i in range(bs)]
                    enc = [arithmeticcoding_fast.ArithmeticEncoder(32, bitout[i]) for i in range(bs)]
                    prob = np.ones(alphabet_size)/alphabet_size
                    cumul = np.zeros(alphabet_size+1, dtype = np.uint64)
                    cumul[1:] = np.cumsum(prob*10000000 + 1)        
                    for i in range(bs):
                            for j in range(min(timesteps, num_iters)):
                                    enc[i].write(cumul, X[ind[i],j])
                    cumul = np.zeros((bs, alphabet_size+1), dtype = np.uint64)
                    for j in (range(num_iters - timesteps)):
                            x=torch.Tensor(X[ind,:])
                            x = x.reshape(-1,timesteps, args.input_size).to(device)
                            outputs = model(x)
                            prob=F.softmax(outputs).data.cpu().numpy()
                            cumul[:,1:] = np.cumsum(prob*10000000 + 1, axis = 1)
                            for i in range(bs):
                                    enc[i].write(cumul[i,:], y_original[ind[i]])
                            ind = ind + 1
                    # close files
                    for i in range(bs):
                            enc[i].finish()
                            bitout[i].close()
                            f[i].close()            
            else:
                    f = open(args.temp_file_prefix+'.last','wb')
                    bitout = arithmeticcoding_fast.BitOutputStream(f)
                    enc = arithmeticcoding_fast.ArithmeticEncoder(32, bitout)
                    prob = np.ones(alphabet_size)/alphabet_size
                    cumul = np.zeros(alphabet_size+1, dtype = np.uint64)
                    cumul[1:] = np.cumsum(prob*10000000 + 1)        

                    for j in range(timesteps):
                            enc.write(cumul, X[0,j])
                    for i in (range(len(X))):
                            x=torch.Tensor(X[i,:])
                            x = x.reshape(-1,timesteps, args.input_size).to(device)
                            outputs = model(x)
                            prob=F.softmax(outputs).data.cpu().numpy()
                            cumul[1:] = np.cumsum(prob*10000000 + 1)
                            enc.write(cumul, y_original[i][0])
                    enc.finish()
                    bitout.close()
                    f.close()
            return


    # variable length integer encoding http://www.codecodex.com/wiki/Variable-Length_Integers
    def var_int_encode(byte_str_len, f):
            while True:
                    this_byte = byte_str_len&127
                    byte_str_len >>= 7
                    if byte_str_len == 0:
                            f.write(struct.pack('B',this_byte))
                            break
                    f.write(struct.pack('B',this_byte|128))
                    byte_str_len -= 1

    # compress the data 
    predict_lstm(X, Y, Y_original, timesteps, batch_size, alphabet_size, args.model_name)

    if l < len(series)-timesteps:
            predict_lstm(X[l:,:], Y[l:,:], Y_original[l:], timesteps, 1, alphabet_size, args.model_name, final_step = True)
    else:
            f = open(args.temp_file_prefix+'.last','wb')
            bitout = arithmeticcoding_fast.BitOutputStream(f)
            enc = arithmeticcoding_fast.ArithmeticEncoder(32, bitout) 
            prob = np.ones(alphabet_size)/alphabet_size
            
            cumul = np.zeros(alphabet_size+1, dtype = np.uint64)
            cumul[1:] = np.cumsum(prob*10000000 + 1)        
            for j in range(l, len(series)):
                    enc.write(cumul, series[j])
            enc.finish()
            bitout.close() 
            f.close()

    # combine files into one file
    f = open(args.output_file_prefix+'.combined','wb')
    for i in range(batch_size):
            f_in = open(args.temp_file_prefix+'.'+str(i),'rb')
            byte_str = f_in.read()
            byte_str_len = len(byte_str)
            var_int_encode(byte_str_len, f)
            f.write(byte_str)
            f_in.close()
    f_in = open(args.temp_file_prefix+'.last','rb')
    byte_str = f_in.read()
    byte_str_len = len(byte_str)
    var_int_encode(byte_str_len, f)
    f.write(byte_str)
    f_in.close()
    f.close()
    shutil.rmtree(args.temp_dir)

def compression(file_path,quantization,Setting):
    device = torch.device('cuda' if (torch.cuda.is_available() and quantization=="no") else 'cpu')      
    args=Setting(file_path,quantization)
    compress_model(args,device)

    compressed_path="../data/compressed/"+args.data_name+"/"+file_path+args.append+".compressed.combined"
    size=os.path.getsize(compressed_path)/1024
    print("Size of compressed data in Deepzip with "+file_path+args.append+": %.2f KB" % size)

    return("Size of compressed data in Deepzip with "+file_path+args.append+": %.2f KB" % size)

def check_size(file_path,Setting):
    args=Setting(file_path,0)
    compressed_path="../data/compressed/"+args.data_name+"/"+file_path+args.append+".compressed.combined"
    size=os.path.getsize(compressed_path)/1024
    print("Size of compressed data in Deepzip with "+file_path+args.append+": %.2f KB" % size)
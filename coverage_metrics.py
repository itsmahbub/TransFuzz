"""
NeuraL Coverage (NLC) computation adapted from:
Yuan et al., "Revisiting Neuron Coverage for DNN Testing: A Layer-Wise and Distribution-Aware Criterion", ICSE 2023.

Original implementation:
- Assumes direct invocation of models via model(inputs)
Extensions in TransFuzz:
- Makes the NLC computation differentiable and exposes its gradient with respect to the input, enabling gradient-guided mutation.
- Generalizes the interface to accept an explicit prediction function, allowing NLC computation for models wrapped behind custom APIs
"""

from tqdm import tqdm
import numpy as np
import torch
import tool


class Coverage:
    def __init__(self, model, layer_size_dict, hyper=None, device=None, unroll=False, inference_func=None, track_grad=True, **kwargs):
        self.device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.model.to(self.device)
        self.layer_size_dict = layer_size_dict
        self.init_variable(hyper, **kwargs)
        self.unroll = unroll
        self.inference_func = inference_func
        self.track_grad = track_grad

    def init_variable(self):
        raise NotImplementedError
        
    def calculate(self):
        raise NotImplementedError

    def coverage(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def build(self, data_loader):
        print('Building is not needed.')

    def assess(self, data_loader):
        for data, *_ in tqdm(data_loader):
            if isinstance(data, tuple):
                data = (data[0].to(self.device), data[1].to(self.device))
            else:
                data = data.to(self.device)
            self.step(data)

    def step(self, data):
        cove_dict = self.calculate(data)
        gain = self.gain(cove_dict)
        if gain is not None:
            self.update(cove_dict, gain)

    def update(self, all_cove_dict, delta=None):
        self.coverage_dict = all_cove_dict
        if delta:
            self.current += delta
        else:
            self.current = self.coverage(all_cove_dict)

    def gain(self, cove_dict_new):
        new_rate = self.coverage(cove_dict_new)
        return new_rate - self.current


class NLC(Coverage):
    def init_variable(self, hyper=None):
        assert hyper is None, 'NLC has no hyper-parameter'
        self.estimator_dict = {}
        self.current = 1
        for (layer_name, layer_size) in self.layer_size_dict.items():
            
            if layer_size[0] > 10_000:
                print(layer_size)
                print(f'Warning: layer {layer_name} has {layer_size[0]} neurons, which may cause high memory usage.')
            self.estimator_dict[layer_name] = tool.Estimator(feature_num=layer_size[0], device=self.device)

    def calculate(self, data):
        stat_dict = {}
        layer_output_dict = tool.get_layer_output(self.model, data, unroll=self.unroll, inference_func=self.inference_func, track_grad=self.track_grad)
        for (layer_name, layer_output) in layer_output_dict.items():
       
            if layer_name not in self.estimator_dict:
                continue
      
            info_dict = self.estimator_dict[layer_name].calculate(layer_output.to(self.device))
            stat_dict[layer_name] = (info_dict['Ave'], info_dict['CoVariance'], info_dict['Amount'])
        return stat_dict

    def update(self, stat_dict, gain=None):
        if gain is None:    
            for i, layer_name in enumerate(stat_dict.keys()):
                (new_Ave, new_CoVariance, new_Amount) = stat_dict[layer_name]
                self.estimator_dict[layer_name].Ave = new_Ave.detach()
                self.estimator_dict[layer_name].CoVariance = new_CoVariance.detach()
                self.estimator_dict[layer_name].Amount = new_Amount.detach()
            self.current = self.coverage(self.estimator_dict)
        else:
            (delta, layer_to_update) = gain
            for layer_name in layer_to_update:
                (new_Ave, new_CoVariance, new_Amount) = stat_dict[layer_name]
                self.estimator_dict[layer_name].Ave = new_Ave.detach()
                self.estimator_dict[layer_name].CoVariance = new_CoVariance.detach()
                self.estimator_dict[layer_name].Amount = new_Amount.detach()
            self.current += delta.detach()

    def coverage(self, stat_dict):
        val = 0
        for i, layer_name in enumerate(stat_dict.keys()):
            CoVariance = stat_dict[layer_name].CoVariance
            val += self.norm(CoVariance)
        return val

    def gain(self, stat_new):
        total = 0
        layer_to_update = []
        for i, layer_name in enumerate(stat_new.keys()):
            (new_Ave, new_CoVar, new_Amt) = stat_new[layer_name]
            value = self.norm(new_CoVar) - self.norm(self.estimator_dict[layer_name].CoVariance)
            if value > 0:
                layer_to_update.append(layer_name)
                total += value
        if total > 0:
            return (total, layer_to_update)
        else:
            return None

    def norm(self, vec, mode='L1', reduction='mean'):
        m = np.prod(vec.size())
        assert mode in ['L1', 'L2']
        assert reduction in ['mean', 'sum']
        if mode == 'L1':
            total = vec.abs().sum()
        elif mode == 'L2':
            total = vec.pow(2).sum().sqrt()
        if reduction == 'mean':
            return total / m
        elif reduction == 'sum':
            return total

    def save(self, path):
        print('Saving recorded NLC in %s...' % path)
        stat_dict = {}
        for layer_name in self.estimator_dict.keys():
            stat_dict[layer_name] = {
                'Ave': self.estimator_dict[layer_name].Ave,
                'CoVariance': self.estimator_dict[layer_name].CoVariance,
                'Amount': self.estimator_dict[layer_name].Amount
            }
        torch.save({'stat': stat_dict}, path)

    def load(self, path):
        print('Loading saved NLC from %s...' % path)
        ckpt = torch.load(path)
        stat_dict = ckpt['stat']
        for layer_name in stat_dict.keys():
            self.estimator_dict[layer_name].Ave = stat_dict[layer_name]['Ave']
            self.estimator_dict[layer_name].CoVariance = stat_dict[layer_name]['CoVariance']
            self.estimator_dict[layer_name].Amount = stat_dict[layer_name]['Amount']


if __name__ == '__main__':
    pass

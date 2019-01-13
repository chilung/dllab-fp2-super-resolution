from torch.autograd import Variable
import torch
import torchvision

print(torch.__version__)
dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
print(dummy_input.shape)

# model_name = '../experiment/edsr_baseline_x2/model/model_best.pt'
model_name = 'edsr.model'

model_edsr = torch.load(model_name)
print(model_edsr)

torch.onnx.export(model_edsr, dummy_input, 'edsr.onnx', export_params=True, verbose=True, training=False)

import os
import time
import onnx

edsr_onnx_filename = 'edsr.onnx'
edsr_onnx_model = onnx.load(edsr_onnx_filename)
onnx.checker.check_model(edsr_onnx_model)
print(onnx.helper.printable_graph(edsr_onnx_model.graph))

import onnx
edsr_onnx_filename = 'edsr.onnx'
edsr_onnx_model = onnx.load(edsr_onnx_filename)

from ngraph_onnx.onnx_importer.importer import import_onnx_model
ng_models = import_onnx_model(edsr_onnx_model)

print(ng_models)

import ngraph as ng
ng_model = ng_models[0]
runtime = ng.runtime(backend_name='CPU')
edsr_ng_model = runtime.computation(ng_model['output'], *ng_model['inputs'])

import numpy as np
picture = np.ones([1, 3, 224, 224], dtype=np.float32)

print(edsr_ng_model(picture, 2))


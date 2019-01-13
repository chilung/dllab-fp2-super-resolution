import torch

import os
import onnx

import tensorrt
import onnx_tensorrt.backend as backend
import numpy as np

torch.set_grad_enabled(False)
lr = torch.randn(1, 3, 768, 1020, dtype=torch.float32, device='cuda:0')

for i in range(10):
    print(i)

    pytorch_model_name = 'edsr.model'
    pytorch_edsr_model = torch.load(pytorch_model_name).cuda()

    edsr_onnx_filename = '{}.onnx'.format(pytorch_model_name)
    # print('Export to onnx model {}'.format(edsr_onnx_filename))
    dummy_input = torch.randn_like(lr, dtype=torch.float32, device='cuda:0')
    torch.onnx.export(pytorch_edsr_model, dummy_input, edsr_onnx_filename, export_params=True, verbose=False, training=False)

    edsr_onnx_model = onnx.load(edsr_onnx_filename)
    # print(onnx.helper.printable_graph(edsr_onnx_model.graph))

    tensorrt_engine = backend.prepare(edsr_onnx_model, device='CUDA:0', max_batch_size=1)
    # # lr_np = lr_np.to(torch.device("cuda:0"))
    # # lr.numpy().astype(np.float32)

    # input_data = np.random.random(size=(1, 3, 768, 1020)).astype(np.float32)
    # # for j in range(1000):
    # sr = tensorrt_engine.run(input_data)[0]
    with torch.no_grad():
        image = lr.cpu().numpy().astype(np.float32)
        sr = tensorrt_engine.run(image)[0]
        sr = torch.from_numpy(sr)

    print('complete {}'.format(i))


import sys, collections
from ai_edge_litert.interpreter import Interpreter
for p in sys.argv[1:]:
    it = Interpreter(model_path=p); it.allocate_tensors()
    print("==", p)
    for d in it.get_input_details(): print(" in ", d['name'], d['shape'], d['dtype'].__name__, d['quantization'])
    for d in it.get_output_details(): print(" out", d['name'], d['shape'], d['dtype'].__name__, d['quantization'])
    ops = collections.Counter(o['op_name'] for o in it._get_ops_details())
    print(" ops", dict(ops))
    # MAC estimate for conv layers
    macs = 0
    td = {t['index']: t for t in it.get_tensor_details()}
    for o in it._get_ops_details():
        if o['op_name'] in ('CONV_2D','DEPTHWISE_CONV_2D','FULLY_CONNECTED'):
            w = td[o['inputs'][1]]['shape']; out = td[o['outputs'][0]]['shape']
            import numpy as np
            if o['op_name']=='CONV_2D': macs += np.prod(out)*w[1]*w[2]*w[3]
            elif o['op_name']=='DEPTHWISE_CONV_2D': macs += np.prod(out)*w[1]*w[2]
            else: macs += np.prod(out)*w[1]
    print(" MMAC %.1f" % (macs/1e6))

import numpy as np
from collections import OrderedDict


def batch_input(graph_inputs, s):
    '''
        graph_inputs: value from graph_input()
        s: slice of batch indices
    '''
    batched_input = {}
    for k, v in graph_inputs.items():
        if isinstance(v, np.ndarray):
            batched_input[k] = v[s]
        else:
            batched_input[k] = v
    return batched_input


def set_graph_kwargs(opt):
    graph_kwargs = dict(lr=opt.learning_rate, walk_type=opt.walk_type, loss=opt.loss)
    # embed or linear
    graph_kwargs['trainEmbed'] = opt.trainEmbed

    attrList = []
    attrTable = OrderedDict()
    if opt.transform == 'face':
        if opt.attrPath is not None:
            with open(opt.attrPath, 'r') as f:
                for i, line in enumerate(f.readlines()):
                    if line.strip():
                        attrList.append(line.strip())
                        attrTable.update({line.strip(): i})
                assert len(attrList) == 40, ' len(attrList) should be 40'
        else:
            attrTable = OrderedDict({
                'daylight': 1, 'night': 2, 'sunrisesunset': 3, 'sunny': 5,
                'clouds': 6, 'fog': 7, 'snow': 9, 'warm': 10, 'cold': 11,
                'beautiful': 13, 'flowers': 14, 'spring': 15, 'summer': 16,
                'autumn': 17, 'winter': 18, 'colorful': 20, 'dark': 24,
                'bright': 25, 'rain': 29, 'boring': 37, 'lush': 39})
            attrList = [i for i in attrTable.keys()]

        if not opt.attrList:
            graph_kwargs['attrList'] = attrList
        else:
            graph_kwargs['attrList'] = opt.attrList.split(',')
        graph_kwargs['attrTable'] = attrTable

    elif opt.transform == 'dsprites':
        attrTable.update({'scale': 0})
        attrTable.update({'x': 1})
        attrTable.update({'y': 2})
        attrTable.update({'posx': 3})
        attrTable.update({'posy': 4})
        if not opt.attrList:
            attrList = ['scale', 'x', 'y', 'posx', 'posy']
            graph_kwargs['attrList'] = attrList
        else:
            attrList = graph_kwargs['attrList'] = opt.attrList.split(',')
        graph_kwargs['attrTable'] = attrTable

    elif opt.transform == 'chair':
        attrTable.update({'x': 0})
        attrTable.update({'y': 1})
        attrList += ['x', 'y']
        graph_kwargs['attrList'] = attrList
        graph_kwargs['attrTable'] = attrTable

    elif opt.transform == 'scene':
        with open(opt.attrPath, 'r') as f:
            for i, line in enumerate(f.readlines()):
                if line.strip():
                    attrList.append(line.strip())
                    attrTable.update({line.strip(): i})
            assert len(attrList) == 40, ' len(attrList) should be 40'
        if not opt.attrList:
            graph_kwargs['attrList'] = attrList
        else:
            graph_kwargs['attrList'] = opt.attrList.split(',')
        graph_kwargs['attrTable'] = attrTable
    elif opt.transform == 'xray':
        # Original attributes
        # attrTable = OrderedDict({"No_Finding": 0,
        #                          "Enlarged_Cardiomediastinum": 1, "Cardiomegaly": 2, "Lung_Opacity": 3,
        #                          "Lung_Lesion": 4, "Edema": 5, "Consolidation": 6, "Pneumonia": 7,
        #                          "Atelectasis": 8, "Pneumothorax": 9, "Pleural_Effusion": 10,
        #                          "Pleural_Other": 11, "Fracture": 12, "Support_Devices": 13})

        # Regressor's attributes
        attrTable = OrderedDict({"Cardiomegaly": 0,
                                 "Edema": 1,
                                 "Consolidation": 2,
                                 "Atelectasis": 3,
                                 "Effusion": 4})

        attrList = [i for i in attrTable.keys()]
        if not opt.attrList:
            graph_kwargs['attrList'] = attrList

        else:
            graph_kwargs['attrList'] = opt.attrList.split(',')
        graph_kwargs['attrTable'] = attrTable

    try:
        graph_kwargs['layers'] = opt.layers.split(',')
    except:
        graph_kwargs['layers'] = None

    if opt.walk_type.startswith('NN'):
        if opt.nn.eps:
            graph_kwargs['eps'] = opt.nn.eps
        if opt.nn.num_steps:
            graph_kwargs['N_f'] = opt.nn.num_steps
    if opt.color.channel is not None and opt.transform.startswith("color"):
        graph_kwargs['channel'] = opt.color.channel
    if 'stylegan' in opt.model:
        graph_kwargs['stylegan_opts'] = opt.stylegan
    if opt.model == 'pggan':
        graph_kwargs['pgan_opts'] = opt.pggan

    return graph_kwargs

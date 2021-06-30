import importlib

def find_model_using_name(model, transform):
    # Given the option --model [modelname] --transform [transform]
    model_filename = "graphs.transform_graph_scene"
    modellib = importlib.import_module(model_filename)

    #graphs = modellib.get_transform_graphs(model.split('_')[0])
    graphs = modellib.get_transform_graphs(model)
    model = None

    target_transform = transform.replace('_', '') + 'graph'
    for g in graphs:
        if g.__name__.lower() == target_transform.lower():
            print('Find NAME: ', target_transform.lower())
            model = g

    if model is None:
        print("In %s.py, there should be a Class with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model


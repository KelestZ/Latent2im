import os
import numpy as np
from options.vis_options import VisOptions
from utils import util
import importlib
import graphs
from utils import html

import torch
import torchvision
import torchvision.transforms as transforms
from collections import OrderedDict
from PIL import Image
from scipy.spatial.distance import cosine


"""
Usage example:

python eval.py models_celeba/stylegan_v2_real_face_linear_lr0.0001_l2_w/opt.yml \
        --gpu 3 --noise_seed 0 --num_samples 10 --num_panels 10 \
        --attrPath ./dataset/attributes_celeba.txt \
        --save_path_w  ./models_celeba/stylegan_v2_real_face_linear_lr0.0001_l2_w/model_w_10_final_walk_module.ckpt \
        --target_attrList Smiling           
"""



def initialize_model():
    from facenet_pytorch import MTCNN, InceptionResnetV1
    resnet = InceptionResnetV1(pretrained='vggface2').cuda().eval()
    return resnet

if __name__ == '__main__':
    v = VisOptions()
    v.parser.add_argument('--num_samples', type=int, default=10,
                          help='number of samples per category')
    v.parser.add_argument('--num_panels', type=int, default=7,
                          help='number of panels to show')
    v.parser.add_argument('--max_alpha', type=float, default=1,
                          help='maximum alpha value')
    v.parser.add_argument('--min_alpha', type=float, default=0,
                          help='minimum alpha value')
    v.parser.add_argument('--layers', type=str, default=None,
                          help='minimum alpha value')
    v.parser.add_argument('--target_attrList', type=str, default=None)

    v.parser.add_argument('--trainEmbed', action='store_true')
    v.parser.add_argument('--updateGAN', action='store_true')

    opt, conf = v.parse()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

    face_rec = initialize_model()
    print('Load face recognition model')

    if opt.output_dir:
        output_dir = opt.output_dir
    else:
        output_dir = os.path.join(conf.output_dir, 'images')
    os.makedirs(output_dir, exist_ok=True)

    graph_kwargs = util.set_graph_kwargs(conf)
    print('Load utils and constants: %s' % conf.model)
    graph_util = importlib.import_module('graphs.' + conf.model + '.graph_util')
    constants = importlib.import_module('graphs.' + conf.model + '.constants')


    print('Find_model_using_name')
    model = graphs.find_model_using_name(conf.model, conf.transform)

    print('Model initialization')
    g = model(**graph_kwargs)
    print('Load multi models')
    if opt.updateGAN:
        g.load_multi_models(opt.save_path_w, opt.save_path_gan, trainEmbed=opt.trainEmbed, updateGAN=opt.updateGAN)
    else:
        g.load_multi_models(opt.save_path_w, None, trainEmbed=opt.trainEmbed, updateGAN=opt.updateGAN)

    num_samples = opt.num_samples
    noise_seed = opt.noise_seed
    batch_size = constants.BATCH_SIZE

    graph_inputs = graph_util.graph_input(g, num_samples, seed=noise_seed)

    # print('Start visualization')

    if 'final' in opt.save_path_w:
        epochs = opt.save_path_w.split('/')[-1].split('_')[2]
    else:
        epochs = opt.save_path_w.split('/')[-1].split('_')[2]

    filename = os.path.join(output_dir, 'w_{}_seed{}'.format(
        epochs, noise_seed))

    name = conf.attrList.strip().split(',')[0]
    print('Name and epochs: ', name, epochs)

    if opt.layers == 'None' or opt.layers == None:
        layers = None
    else:
        layers = [int(i) for i in opt.layers.split(',')]

    # 256*256 resolution
    step = 6

    # Evaluation
    attrList = []
    attrTable = OrderedDict()

    with open(opt.attrPath, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if line.strip():
                attrList.append(line.strip())
                attrTable.update({line.strip(): i})
        assert len(attrList) == 40, ' len(attrList) should be 40'

    if not conf.attrList:
        opt.attrList = attrList
    else:
        opt.attrList = conf.attrList.split(',')
    opt.attrTable = attrTable

    if not opt.target_attrList:
        target_attrList = opt.attrList
    else:
        target_attrList = opt.target_attrList.strip().split(',')
    print('target_attrList: ', target_attrList)


    #####
    # For evaluation
    multi_attrs = [[], [], []]
    original_attrs = [[], [], []]
    embeddings = []
    sim = [[], [], []]

    for batch_start in range(0, num_samples, batch_size):
        s = slice(batch_start, min(num_samples, batch_start + batch_size))
        graph_inputs_batch = util.batch_input(graph_inputs, s)

        max_alpha = opt.max_alpha
        min_alpha = opt.min_alpha

        new_filename = filename + '_{}_max{}_min{}'.format(name, max_alpha, min_alpha)


        alphas_to_graph, alphas_to_target = g.vis_image_batch(graph_inputs_batch, new_filename, s.start,
                                                                  num_panels=opt.num_panels, max_alpha=max_alpha,
                                                                  min_alpha=min_alpha, wgt=True)

        for count, i in enumerate(range(len(target_attrList))):
            index_ = opt.attrTable[target_attrList[i]]
            key = list(opt.attrTable.keys())[index_]
            new_filename2 = new_filename + '_attr_%s' % key
            # print('Save result in %s ' % new_filename2)
            # alphas_to_graph: list(num_panel) -> [B, C], alphas_to_target: list(num_panel) -> scalar

            #########  Inference only  #########
            # g.vis_multi_image_batch_alphas(graph_inputs_batch, new_filename2,
            #                                alphas_to_graph=alphas_to_graph,
            #                                alphas_to_target=alphas_to_target,
            #                                layers=layers,
            #                                batch_start=s.start,
            #                                wgt=False, wmask=False,
            #                                trainEmbed=opt.trainEmbed, computeL2=False,
            #                                index_=index_)  # , given_w=given_w)


            #########  Evaluation  #########
            multi_attr, org_attr, imgs, org_img = g.vis_multi_image_batch_alphas_compute_multi_attr(graph_inputs_batch, new_filename2,
                                                                   alphas_to_graph=alphas_to_graph,
                                                                   alphas_to_target=alphas_to_target,
                                                                   layers=layers,
                                                                   batch_start=s.start,
                                                                   wgt=False, wmask=False,
                                                                   trainEmbed=opt.trainEmbed, computeL2=False,
                                                                   index_=index_)

            for k in range(3):
                for i in range(len(imgs[k])):

                    org = Image.fromarray(np.transpose(org_img[k][i], (1,2,0)))
                    reshaped_org = org.resize((160, 160))
                    reshaped_org = torch.Tensor(np.transpose(np.array(reshaped_org), (2,0,1))).cuda().unsqueeze(0)

                    # Compute the embedding of the original image
                    embed_org = face_rec(reshaped_org)
                    img = Image.fromarray(np.transpose(imgs[k][i], (1, 2, 0)))
                    reshaped_img = img.resize((160, 160))
                    reshaped_img = torch.Tensor(np.transpose(np.array(reshaped_img), (2, 0, 1))).cuda().unsqueeze(0)
                    # Compute the embedding of the edited image
                    embed = face_rec(reshaped_img)
                    # Compute the Cosine similarity for image identity preservation
                    similarity = cosine(embed.detach().cpu().numpy(), embed_org.detach().cpu().numpy())
                    sim[k].append(similarity)

    results_avg = []
    results = []
    for k in range(3):
        # print('k len: ', k, len(sim[k]))
        if len(sim[k]) == 0:
            continue
        result = np.sum(sim[k])
        result_avg = 1-np.mean(sim[k])
        results.append(result)
        results_avg.append(result_avg)
    print('[IDENTITY PRESERVATION] Results on 3 epsilon segments', ['%.4f' % i for i in results_avg])

    multi_attrs[0] += multi_attr[0]
    multi_attrs[1] += multi_attr[1]
    multi_attrs[2] += multi_attr[2]

    original_attrs[0] += org_attr[0]
    original_attrs[1] += org_attr[1]
    original_attrs[2] += org_attr[2]

    for k in range(3):
        # print(k, len(multi_attrs[k]))
        # print(k, len(original_attrs[k]))
        multi_attrs[k] = np.array(multi_attrs[k])
        original_attrs[k] = np.array(original_attrs[k])
        # print(k, multi_attrs[k].shape, original_attrs[k].shape)

    results = []
    results_avg = []

    for k in range(3):
        if (original_attrs[k].shape[0] == 0):
            continue

        org = np.hstack([original_attrs[k][:, :int(index_)], original_attrs[k][:, int(index_ + 1):]])
        changed = np.hstack([multi_attrs[k][:, :int(index_)], multi_attrs[k][:, int(index_ + 1):]])
        result = np.sum(np.abs(changed - org))
        result_avg = np.mean(np.abs(changed - org))
        results.append(result)
        results_avg.append(result_avg)

    print('[ATTRIBUTE PRESERVATION] Results on 3 epsilon segments', ['%.4f' % i for i in results_avg])


# html.make_html(output_dir)





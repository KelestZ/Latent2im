import os
import time
import math
import torch
import graphs
import importlib
import numpy as np
import logging
import utils.logging
from utils import util, image
from options.train_options import TrainOptions
from torch.utils.tensorboard import SummaryWriter

"""
Usage: 
# face
python train.py --model stylegan_v2_real --transform face \
        --num_samples 20000 --learning_rate 1e-4 --latent w \
        --walk_type linear --loss l2 --gpu 3 --attrList Smiling \
        --attrPath './dataset/attributes_celeba.txt' \
        --models_dir ./models_celeba --overwrite_config 
"""


def train(graphs, graph_inputs, output_dir, attrList,
          layers=None, save_freq=100, trainEmbed=False,
          updateGAN=False, opt=None):
    # configure logging file
    logging_file = os.path.join(output_dir, 'log.txt')
    if not os.path.exists(output_dir + '/logs/'):
        os.mkdir(output_dir + '/logs/')
    writer = SummaryWriter(output_dir + '/logs/')
    utils.logging.configure(logging_file, append=False)
    n_epoch = 10

    batch_size = constants.BATCH_SIZE
    num_samples = graph_inputs['z'].shape[0]

    for epoch in range(n_epoch):
        if updateGAN:
            raise ('ERROR: jointly training is not implemented yet')
        else:
            ITERS = (num_samples // batch_size)

        graph_inputs = graph_util.graph_input(graphs, num_samples, seed=epoch)
        print('Number of the training epochs and iterations: ', n_epoch, ITERS)

        for i in range(ITERS):
            batch_start = i * batch_size
            start_time = time.time()
            s = slice(batch_start, min(num_samples, batch_start + batch_size))
            graph_inputs_batch = util.batch_input(graph_inputs, s)

            zs_batch = graph_inputs_batch['z']
            graph_inputs_batch_cuda = {}
            graph_inputs_batch_cuda['z'] = torch.Tensor(graph_inputs_batch['z']).cuda()

            # multi-attribute transformation
            z_global = graph_inputs_batch_cuda['z']

            # get w = MLP(z)
            w_global = graphs.get_w(z_global)
            graph_inputs_batch_cuda['w'] = w_global

            # get img = netG(w)
            out_zs = graphs.get_logits(graph_inputs_batch_cuda)

            # get regression preds alpha = R(I_fake) -> N, C
            alpha_org = graphs.get_reg_preds(out_zs)

            alphas_reg = []

            # alpha_for_graph: N x len(attrList), alpha_for_target: len(attrList) (numpy)
            alpha_for_graph, alpha_for_target, index_embed = graphs.get_train_alpha(zs_batch,
                                                                       N_attr=len(attrList),
                                                                       trainEmbed=trainEmbed)

            alphas_reg.append(alpha_for_graph)

            if not isinstance(alpha_for_graph, list):
                alpha_for_graph = [alpha_for_graph]
                alpha_for_target = [alpha_for_target]

            for ag, at in zip(alpha_for_graph, alpha_for_target):
                ag = torch.tensor(ag).float().cuda()
                epsilon = graphs.get_alphas(alpha_org, ag)

                # w = w + eT
                w_new = graphs.get_w_new_tensor(w_global, epsilon,
                                                layers=layers)

                transformed_inputs = graph_inputs_batch_cuda
                transformed_inputs['w'] = w_new
                transformed_output = graphs.get_logits(transformed_inputs)
                w_global = w_new

                feed_dict = {}
                feed_dict['w'] = w_global
                feed_dict['org'] = out_zs
                feed_dict['logit'] = transformed_output
                feed_dict['alpha'] = ag

            curr_loss = graphs.optimizeParametersAll(feed_dict,
                                                     trainEmbed=trainEmbed,
                                                     updateGAN=updateGAN,
                                                     no_content_loss=opt.no_content_loss,
                                                     no_gan_loss=opt.no_gan_loss
                                                     )

            curr_loss_item = curr_loss.detach().cpu().item()
            writer.add_scalar('Loss/train', curr_loss_item, epoch*ITERS+i)

            elapsed_time = time.time() - start_time

            logging.info('T, epc, bst, lss, alpha: {}, {}, {}, {}, {}'.format(
                elapsed_time, epoch, batch_start, curr_loss, round(at[0], 2)))

            if (i % save_freq == 0):
                make_samples(out_zs, output_dir, epoch, i * batch_size, batch_size,
                             name='org_%.2f' % (round(at[0], 2)))
                make_samples(transformed_output, output_dir, epoch, i * batch_size, batch_size,
                             name='logit_%.2f' % (round(at[0], 2)))

        graphs.save_multi_models('{}/model_w_{}'.format(output_dir, epoch),
                                 '{}/model_gan_{}.ckpt'.format(output_dir, epoch),
                                 trainEmbed=trainEmbed,
                                 updateGAN=updateGAN)

    graphs.save_multi_models('{}/model_w_{}_final'.format(output_dir, n_epoch),
                             '{}/model_gan_{}_final.ckpt'.format(output_dir, n_epoch),
                             trainEmbed=trainEmbed,
                             updateGAN=updateGAN)

    writer.close()


def make_samples(img_tensor, output_dir, epoch, optim_iter, batch_size, pre_path='results', name='test'):
    if img_tensor.is_cuda:
        img_tensor = img_tensor.detach().cpu().numpy()
    img_tensor = np.uint8(np.clip(((img_tensor + 1) / 2.0) * 255, 0, 255))
    if img_tensor.shape[1] == 1 or img_tensor.shape[1] == 3:
        img_tensor = np.transpose(img_tensor, [0, 2, 3, 1])
    image.save_im(image.imgrid(img_tensor, cols=int(math.sqrt(batch_size))),
                  '{}/{}/{}_{}_{}'.format(output_dir, pre_path, epoch, optim_iter, name))


if __name__ == '__main__':

    opt = TrainOptions().parse()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

    output_dir = opt.output_dir
    if not os.path.exists(os.path.join(output_dir, 'results')):
        os.makedirs(os.path.join(output_dir, 'results'))

    # set attrTable
    graph_kwargs = util.set_graph_kwargs(opt)

    graph_util = importlib.import_module('graphs.' + opt.model + '.graph_util')
    constants = importlib.import_module('graphs.' + opt.model + '.constants')
    model = graphs.find_model_using_name(opt.model, opt.transform)

    g = model(**graph_kwargs)

    num_samples = opt.num_samples
    graph_inputs = graph_util.graph_input(g, num_samples, seed=0)

    if opt.suffix:
        name = opt.suffix
    else:
        name = None

    attrList = graph_kwargs['attrList']
    layers = opt.layers

    print('attrlist: ', attrList)

    train(g, graph_inputs, output_dir,
          attrList,
          layers=layers,
          save_freq=opt.model_save_freq,
          trainEmbed=opt.trainEmbed,
          updateGAN=opt.updateGAN,
          opt=opt
          )

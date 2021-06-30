import numpy as np
import os
import pickle
from . import constants
from .graph_util import *
# from resources import tf_lpips_pkg as lpips_tf
import torch, torchvision
from utils import image

import torch.nn as nn
from torch.autograd import Variable


import sys
sys.path.append('resources/progressive_growing_of_gans')

class TransformGraph():
    def __init__(self, lr, walk_type, nsliders, loss_type, eps, N_f,
                 pgan_opts):

        assert(loss_type in ['l2', 'lpips']), 'unimplemented loss'

        # module inputs
        self.useGPU = constants.useGPU
        self.module = self.get_pgan_module()
        self.LAMBDA = 10
        # TODO: CPU VERSION
        self.dim_z = constants.DIM_Z
        self.Nsliders = Nsliders = nsliders
        self.img_size = constants.resolution
        self.num_channels = constants.NUM_CHANNELS
        self.lr = lr
        self.BCE_loss = nn.BCELoss()
        self.BCE_loss_logits = nn.BCEWithLogitsLoss()
        # walk pattern
        scope = 'walk'
        if walk_type == 'NNz':
            with tf.name_scope(scope):
                # alpha is the integer number of steps to take
                alpha = tf.placeholder(tf.int32, shape=())
                T1 = tf.keras.layers.Dense(dim_z, input_shape=(None, dim_z), kernel_initializer='RandomNormal', bias_initializer='zeros', activation=tf.nn.relu)
                T2 = tf.keras.layers.Dense(dim_z, input_shape=(None, dim_z), kernel_initializer='RandomNormal', bias_initializer='zeros')
                T3 = tf.keras.layers.Dense(dim_z, input_shape=(None, dim_z), kernel_initializer='RandomNormal', bias_initializer='zeros', activation=tf.nn.relu)
                T4 = tf.keras.layers.Dense(dim_z, input_shape=(None, dim_z), kernel_initializer='RandomNormal', bias_initializer='zeros')
                # forward transformation
                out_f = []
                z_prev = z
                z_norm = tf.norm(z, axis=1, keepdims=True)
                for i in range(1, N_f + 1):
                    z_step = z_prev + T2(T1(z_prev))
                    z_step_norm = tf.norm(z_step, axis=1, keepdims=True)
                    z_step = z_step * z_norm / z_step_norm
                    out_f.append(z_step)
                    z_prev = z_step

                # reverse transformation
                out_g = []
                z_prev = z
                z_norm = tf.norm(z, axis=1, keepdims=True)
                for i in range(1, N_f + 1):
                    z_step = z_prev + T4(T3(z_prev))
                    z_step_norm = tf.norm(z_step, axis=1, keepdims=True)
                    z_step = z_step * z_norm / z_step_norm
                    out_g.append(z_step)
                    z_prev = z_step
                out_g.reverse() # flip the reverse transformation

                # w has shape (2*N_f + 1, batch_size, dim_z)
                # elements 0 to N_f are the reverse transformation, in reverse order
                # elements N_f + 1 to 2*N_f + 1 are the forward transformation
                # element N_f is no transformation
                w = tf.stack(out_g+[z]+out_f)
        elif walk_type == 'linear':
            # TODO: DOES 0-0.1 INITIALIZATION GOOD?
            w = torch.Tensor(np.random.normal(0.0, 0.1, [1, self.dim_z, Nsliders]))
            self.w = w.cuda()
            self.w.requires_grad_()
        else:
            raise NotImplementedError('Not implemented walk type:' '{}'.format(walk_type))

        self.optimizer = train_optimizer = torch.optim.Adam([self.w], lr=self.lr, betas=(0.9, 0.99))

        # set class vars
        self.Nsliders = Nsliders

        self.y = None
        self.z = None
        self.truncation = None

        self.walk_type = walk_type
        self.N_f = N_f  # NN num_steps
        self.eps = eps  # NN step_size

    def get_logits(self, inputs_dict):
        outputs_orig = self.module.netG(inputs_dict['z'])
        return outputs_orig

    def get_z_new(self, z, alpha):
        if self.walk_type == 'linear' or self.walk_type == 'NNz':
            for i in range(self.Nsliders):
                # TODO: PROBLEM HERE
                al = torch.unsqueeze(torch.Tensor(alpha[:, i]), axis=1)
                z_new = (z + al * self.w[:, :, i]).cuda()
        return z_new

    def get_z_new_tensor(self, z, alpha):
        if self.walk_type == 'linear' or self.walk_type == 'NNz':
            # print('Nsliders: ', self.Nsliders)
            for i in range(self.Nsliders):
                al = torch.unsqueeze(alpha[:, i], axis=1)
                z_new = z.cpu() + al.cpu() * self.w[:, :, i].cpu()

        return z_new.cuda()

    def get_loss(self, feed_dict):
        # L2 loss
        target = feed_dict['target']
        mask = feed_dict['mask_out']
        logit = feed_dict['logit']
        diff = (logit - target) * mask
        return torch.mean(diff.pow(2))

    def get_edit_loss(self, feed_dict):
        # L2 loss
        target = feed_dict['target']
        mask = feed_dict['mask_out']
        logit = feed_dict['logit']
        diff = (logit - target) * mask
        return torch.mean(diff.pow(2))

    def update_params(self, loss):
        # print('Before w: ', self.w[0, :3])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        # print('Update w: ', self.w[0, :3])

    def save_model(self, save_path_w, save_path_gan):
        print('Save W and GAN in %s' % save_path_w)
        # torch.save(self.module.state_dict(), save_path_gan)
        self.module.save(save_path_gan)
        np.save(save_path_w, self.w.detach().cpu().numpy())

    def load_model(self, save_path_w, save_path_gan):

        # Load w
        print('Load W in %s' % save_path_w)
        print('Before w: ', self.w[0, :5, 0])
        self.w = torch.Tensor(np.load(save_path_w))
        print('After w: ', self.w[0, :5, 0])
        # Load GAN
        print('Load GAN in %s' % save_path_gan)
        self.module.load(save_path_gan)


    def get_pgan_module(self):
        print('Loading PGGAN module')
        # They only trained on fashion data
        # 128*128
        module = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 'PGAN',
                                pretrained=True,
                                model_name='celeba',
                                useGPU=self.useGPU)
        return module

    def optimizeParameters(self, feed_dict):
        # target = feed_dict['target']
        mask = feed_dict['mask_out']
        logit = feed_dict['logit']
        x_real = feed_dict['real_target']

        y_real = Variable(torch.ones(logit.size()[0]).cuda())
        y_fake = Variable(torch.zeros(logit.size()[0]).cuda())

        # Update D
        self.module.optimizerD.zero_grad()
        D_real_result = self.module.netD(x_real).squeeze()
        # print(D_real_result)

        D_real_loss = self.BCE_loss_logits(D_real_result, y_real)
        D_fake_result = self.module.netD(logit).squeeze()
        D_fake_loss = self.BCE_loss_logits(D_fake_result, y_fake)
        D_train_loss = D_real_loss + D_fake_loss
        # print('D_train_loss: ', D_train_loss)
        D_train_loss.backward(retain_graph=True)
        self.module.optimizerD.step()

        # Update G
        self.module.optimizerG.zero_grad()
        new_logit = self.get_logits(feed_dict)
        feed_dict['logit'] = new_logit

        D_fake_result = self.module.netD(new_logit).squeeze()
        G_train_loss = self.BCE_loss_logits(D_fake_result, y_real)
        Edit_loss = self.get_edit_loss(feed_dict)
        G_train_loss += self.LAMBDA * Edit_loss
        # print('G_train_loss: ', G_train_loss)
        G_train_loss.backward(retain_graph=True)
        self.module.optimizerG.step()

        # Update w
        self.optimizer.zero_grad()
        new_logit = self.get_logits(feed_dict)
        feed_dict['logit'] = new_logit
        Edit_loss = self.get_edit_loss(feed_dict)
        Edit_loss.backward()
        self.optimizer.step()

        return Edit_loss


    def clip_ims(self, ims):
        return np.clip(np.rint((ims + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8)

    def clip_ims_sig(self, ims):
        return np.clip(np.rint((ims * 255.0), 0.0, 255.0).astype(np.uint8)

    def apply_alpha(self, graph_inputs, alpha_to_graph):

        zs_batch = graph_inputs['z']  # tensor.cuda() # [Batch, DIM_Z]

        # print(alpha_to_graph)
        if self.walk_type == 'linear':
            # best_inputs = {self.z: zs_batch, self.alpha: alpha_to_graph}
            # best_im_out = self.sess.run(self.transformed_output, best_inputs)

            alpha_to_graph = torch.tensor(alpha_to_graph).float().cuda()
            z_new = self.get_z_new_tensor(zs_batch, alpha_to_graph)
            best_inputs = {'z': z_new}

            best_im_out = self.get_logits(best_inputs)

            return best_im_out
        elif self.walk_type == 'NNz':
            # alpha_to_graph is number of steps and direction
            direction = np.sign(alpha_to_graph)
            num_steps = np.abs(alpha_to_graph)
            # embed()
            single_step_alpha = self.N_f + direction
            # within the graph range, we can compute it directly
            if 0 <= alpha_to_graph + self.N_f <= self.N_f * 2:
                zs_out = self.sess.run(self.z_new, {
                    self.z:zs_batch, self.alpha:
                    alpha_to_graph + self.N_f})
                # # sanity check
                # zs_next = zs_batch
                # for n in range(num_steps):
                #     feed_dict = {self.z: zs_next, self.alpha: single_step_alpha}
                #     zs_next = self.sess.run(self.z_new, feed_dict=feed_dict)
                # zs_test = zs_next
                # assert(np.allclose(zs_test, zs_out))
            else:
                # print("recursive zs for {} steps".format(alpha_to_graph))
                zs_next = zs_batch
                for n in range(num_steps):
                    feed_dict = {self.z: zs_next, self.alpha: single_step_alpha}
                    zs_next = self.sess.run(self.z_new, feed_dict=feed_dict)
                zs_out = zs_next
            # already taken n steps at this point, so directly use zs_next
            # without any further modifications: using self.N_f index into w
            # alternatively, could also use self.outputs_orig
            best_inputs = {self.z: zs_out, self.labels: labels_batch,
                           self.alpha: self.N_f}

            best_im_out = self.sess.run(self.transformed_output,
                                        best_inputs)
            return best_im_out

    def vis_image_batch_alphas(self, graph_inputs, filename,
                               alphas_to_graph, alphas_to_target,
                               batch_start, wgt=False, wmask=False):

        zs_batch = graph_inputs['z']  # numpy

        filename_base = filename
        ims_target = []
        ims_transformed = []
        ims_mask = []
        for ag, at in zip(alphas_to_graph, alphas_to_target):
            input_test = {'z': torch.Tensor(zs_batch).cuda()}
            out_input_test = self.get_logits(input_test)
            out_input_test = out_input_test.detach().cpu().numpy()  # on Cuda
            target_fn, mask_out = self.get_target_np(out_input_test, at)

            best_im_out = self.apply_alpha(input_test, ag).detach().cpu().numpy()

            ims_target.append(target_fn)
            ims_transformed.append(best_im_out)
            ims_mask.append(mask_out)

        for ii in range(zs_batch.shape[0]):

            arr_gt = np.stack([x[ii, :, :, :] for x in ims_target], axis=0)
            if wmask:
                arr_transform = np.stack([x[j, :, :, :] * y[j, :, :, :] for x, y
                                          in zip(ims_transformed, ims_mask)], axis=0)
            else:
                arr_transform = np.stack([x[ii, :, :, :] for x in
                                          ims_transformed], axis=0)
            arr_gt = self.clip_ims(arr_gt)
            arr_transform = self.clip_ims(arr_transform)
            if wgt:
                ims = np.concatenate((arr_gt, arr_transform), axis=0)
            else:
                ims = arr_transform
            filename = filename_base + '_sample{}'.format(ii + batch_start)
            if wgt:
                filename += '_wgt'
            if wmask:
                filename += '_wmask'
            # (7, 3, 64, 64)
            if ims.shape[1] == self.num_channels:
                # N C W H -> N W H C
                ims = np.transpose(ims, [0, 2, 3, 1])

            image.save_im(image.imgrid(ims, cols=len(alphas_to_graph)), filename)

    def vis_image_batch(self, graph_inputs, filename,
                        batch_start, wgt=False, wmask=False, num_panels=7):
        raise NotImplementedError('Subclass should implement vis_image_batch')


class PixelTransform(TransformGraph):
    def __init__(self, *args, **kwargs):
        TransformGraph.__init__(self, *args, **kwargs)

    def get_distribution_statistic(self, img, channel=None):
        raise NotImplementedError('Subclass should implement get_distribution_statistic')

    def get_distribution(self, num_samples, channel=None):
        random_seed = 0
        rnd = np.random.RandomState(random_seed)
        inputs = graph_input(self, num_samples, seed=random_seed)
        batch_size = constants.BATCH_SIZE
        model_samples = []
        for a in self.test_alphas():
            distribution = []
            start = time.time()
            print("Computing attribute statistic for alpha={:0.2f}".format(a))
            for batch_num, batch_start in enumerate(range(0, num_samples,
                                                          batch_size)):
                s = slice(batch_start, min(num_samples, batch_start + batch_size))
                inputs_batch = util.batch_input(inputs, s)
                zs_batch = inputs_batch[self.z]
                a_graph = self.scale_test_alpha_for_graph(a, zs_batch, channel)
                ims = self.clip_ims(self.apply_alpha(inputs_batch, a_graph))
                for img in ims:
                    img_stat = self.get_distribution_statistic(img, channel)
                    distribution.extend(img_stat)
            end = time.time()
            print("Sampled {} images in {:0.2f} min".format(num_samples, (end-start)/60))
            model_samples.append(distribution)

        model_samples = np.array(model_samples)
        return model_samples



class BboxTransform(TransformGraph):
    def __init__(self, *args, **kwargs):
        TransformGraph.__init__(self, *args, **kwargs)

    def get_distribution_statistic(self, img, channel=None):
        raise NotImplementedError('Subclass should implement get_distribution_statistic')

    def get_distribution(self, num_samples, **kwargs):
        if 'is_face' in self.dataset:
            from utils.detectors import FaceDetector
            self.detector = FaceDetector()
        elif self.dataset['coco_id'] is not None:
            from utils.detectors import ObjectDetector
            self.detector = ObjectDetector(self.sess)
        else:
            raise NotImplementedError('Unknown detector option')

        # not used for faces: class_id=None
        class_id = self.dataset['coco_id']

        random_seed = 0
        rnd = np.random.RandomState(random_seed)

        model_samples = []
        for a in self.test_alphas():
            distribution = []
            total_count = 0
            start = time.time()
            print("Computing attribute statistic for alpha={:0.2f}".format(a))
            while len(distribution) < num_samples:
                inputs = graph_input(self, 1, seed=total_count)
                zs_batch = inputs[self.z]
                a_graph = self.scale_test_alpha_for_graph(a, zs_batch)
                ims = self.clip_ims(self.apply_alpha(inputs, a_graph))
                img = ims[0, :, :, :]
                img_stat = self.get_distribution_statistic(img, class_id)
                if len(img_stat) == 1:
                    distribution.extend(img_stat)
                total_count += 1
            end = time.time()
            print("Sampled {} images to detect {} boxes in {:0.2f} min".format(
                total_count, num_samples, (end-start)/60))
            model_samples.append(distribution)

        model_samples = np.array(model_samples)
        return model_samples

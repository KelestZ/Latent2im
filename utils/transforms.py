import numpy as np
import cv2
from skimage.color import rgb2lab, lab2rgb
from .rotate3d import ImageTransformer
import torch
# from .relgan.relgan_model import relgan

class ColorTransform():
    def __init__(self, channel=None):
        # channel to modify, use None for all channels (linear) or luminance (NN)
        self.channel=channel
        self.alpha_original = 0
        self.max_x = 255

    def get_target_np(self, outputs_zs, alpha):
        ''' return target image and mask '''
        mask_out = np.ones(outputs_zs.shape)
        if not np.any(alpha): # alpha is all zeros
            return outputs_zs, mask_out
        target_fn = np.copy(outputs_zs)
        for b in range(alpha.shape[0]):
            for i in range(self.num_channels):
                target_fn[b,:,:,i] = target_fn[b,:,:,i]+alpha[b,i]
        return target_fn, mask_out

    def get_train_alpha(self, zs_batch):
        ''' get an alpha for training, return in format
            alpha_val_for_graph, alpha_val_for get_target_np'''
        batch_size = zs_batch.shape[0]
        if self.walk_type == 'linear':
            if self.channel is None:
                alpha_val = np.random.random(size=(batch_size, self.num_channels))-0.5
            else:
                alpha_val = np.zeros((batch_size, self.num_channels))
                alpha_val[:, self.channel] = np.random.random(size=(batch_size,)) - 0.5
            # graph and target use the same slider value
            alpha_val_for_graph = alpha_val
            alpha_val_for_target = alpha_val
            return alpha_val_for_graph, alpha_val_for_target
        elif self.walk_type == 'NNz':
            alpha_val = np.random.randint(-self.N_f, self.N_f+1)
            if self.channel is None:
                alpha_val_for_target = (alpha_val * self.eps *
                                       np.ones((batch_size, self.num_channels)))
            else:
                alpha_val_for_target = np.zeros((batch_size, self.num_channels))
                alpha_val_for_target[:, self.channel] = alpha_val * self.eps
            alpha_val_for_graph = alpha_val + self.N_f
            return alpha_val_for_graph, alpha_val_for_target

    def scale_test_alpha_for_graph(self, alpha, zs_batch, channel=None,
                                   contrast=False):
        ''' map a scalar alpha to the appropriate shape,
            and do the desired transformation '''
        batch_size = zs_batch.shape[0]
        if self.walk_type == 'linear':
            if channel is None:
                slider = alpha * np.ones((batch_size, self.num_channels))
            else:
                assert(channel >= 0)
                assert(channel < self.num_channels)
                if contrast: # contrast is good for vis
                    slider = -alpha * np.ones((batch_size, self.Nsliders))
                else: # good for channel-specific effects
                    slider = np.zeros((batch_size, self.Nsliders))
                slider[:, channel] = alpha
            return slider
        elif self.walk_type == 'NNz':
            return int(np.round(alpha / self.eps))

    def test_alphas(self):
        return np.linspace(-1, 1, 9)

    def vis_alphas(self, num_panels):
        # alpha range for visualization
        # return np.array([0, 1.0])
        # return np.array([0, 0.5])
        return np.linspace(0, 1, num_panels)

class ColorLabTransform(ColorTransform):

    def get_target_np(self, outputs_zs, alpha):
        ''' return target image and mask '''
        mask_out = np.ones(outputs_zs.shape)
        if not np.any(alpha): # alpha is all zeros
            return outputs_zs, mask_out
        target_fn = np.copy(outputs_zs)
        scaled_alpha = np.copy(alpha)
        # we assume alpha in [-1,1]
        # alpha_L in [-50,50], alpha_a in [-128,128], alpha_b in [-128, 128]
        for b in range(alpha.shape[0]):
            scaled_alpha[b,0] = alpha[b,0] * 50
            scaled_alpha[b,1] = alpha[b,1] * 128
            scaled_alpha[b,2] = alpha[b,2] * 128

        for b in range(alpha.shape[0]):
            target_fn[b,:,:,:] = rgb2lab((target_fn[b,:,:,:]+1)/2)
            for i in range(self.num_channels):
                target_fn[b,:,:,i] = target_fn[b,:,:,i]+scaled_alpha[b,i]
            target_fn[b,:,:,:] = lab2rgb(target_fn[b,:,:,:])*2 - 1
        return target_fn, mask_out

    def get_train_alpha(self, zs_batch):
        ''' get an alpha for training, return in format
            alpha_val_for_graph, alpha_val_for get_target_np'''
        batch_size = zs_batch.shape[0]
        if self.walk_type == 'linear':
            if self.channel is None:
                alpha_val = np.random.random(size=(batch_size, self.num_channels))-0.5
            else:
                alpha_val = np.zeros((batch_size, self.num_channels))
                alpha_val[:, self.channel] = np.random.random(size=(batch_size,)) - 0.5
            # graph and target use the same slider value
            alpha_val_for_graph = alpha_val
            alpha_val_for_target = alpha_val
            return alpha_val_for_graph, alpha_val_for_target
        elif self.walk_type == 'NNz':
            alpha_val = np.random.randint(-self.N_f, self.N_f+1)
            if self.channel is None:
                # adjust luminance
                alpha_val_for_target = np.zeros((batch_size, 3))
                alpha_val_for_target[:, 0] = alpha_val * self.eps
            else:
                alpha_val_for_target = np.zeros((batch_size, 3))
                alpha_val_for_target[:, self.channel] = alpha_val * self.eps
            alpha_val_for_graph = alpha_val + self.N_f
            return alpha_val_for_graph, alpha_val_for_target

    def vis_alphas(self, num_panels):
        # alpha range for visualization
        alphas = np.linspace(-1, 1, num_panels)

class ZoomTransform():

    def __init__(self):
            self.alpha_original = 1
            self.max_x = self.img_size ** 2
            ### NEW
            self.num_panel = 15
            # self.embedding_alpha = np.linspace(0.5, 2.0, self.num_panel)
            #MNIST
            self.embedding_alpha = np.exp(np.linspace(np.log(0.5), np.log(2), self.num_panel))
            #Celeba
            #self.embedding_alpha = np.exp(np.linspace(np.log(0.5), np.log(1.5), self.num_panel))

    def get_target_np(self, outputs_zs, alpha):
        ''' return target image and mask '''
        flag = False
        if outputs_zs.shape[1] == 3 or outputs_zs.shape[1] == 1:
            flag = True
            outputs_zs = np.transpose(outputs_zs, [0, 2, 3, 1])

        img_size = outputs_zs.shape[1]
        mask_fn = np.ones(outputs_zs.shape)

        if alpha == 1:
            if flag == True:
                outputs_zs = np.transpose(outputs_zs, [0, 3, 1, 2])
                mask_fn = np.transpose(mask_fn, [0, 3, 1, 2])
            return outputs_zs, mask_fn

        new_size = int(alpha*img_size)
        ## crop - zoom in
        if alpha < 1:
            output_cropped = outputs_zs[:,img_size//2-new_size//2:img_size//2+new_size//2,
                    img_size//2-new_size//2:img_size//2+new_size//2,:]
            mask_cropped = mask_fn
        ## padding - zoom out
        else:
            output_cropped = np.zeros((outputs_zs.shape[0], new_size, new_size,
                outputs_zs.shape[3]))
            mask_cropped = np.zeros((outputs_zs.shape[0], new_size, new_size,
                outputs_zs.shape[3]))
            output_cropped[:, new_size//2-img_size//2:new_size//2+img_size//2,
                    new_size//2-img_size//2:new_size//2+img_size//2,:] = outputs_zs
            mask_cropped[:, new_size//2-img_size//2:new_size//2+img_size//2,
                    new_size//2-img_size//2:new_size//2+img_size//2,:] = mask_fn

        ## Resize
        target_fn = np.zeros(outputs_zs.shape)
        mask_out = np.zeros(outputs_zs.shape)

        if outputs_zs.shape[-1] == 1:
            # grayscale
            for i in range(outputs_zs.shape[0]):
                target_fn[i, :, :, 0] = cv2.resize(output_cropped[i, :, :, 0], (img_size, img_size),
                                                   interpolation=cv2.INTER_LINEAR)
                mask_out[i, :, :, 0] = cv2.resize(mask_cropped[i, :, :, 0], (img_size, img_size),
                                                  interpolation=cv2.INTER_LINEAR)
        else:
            # rgb
            for i in range(outputs_zs.shape[0]):
                target_fn[i, :, :, :] = cv2.resize(output_cropped[i, :, :, :], (img_size, img_size),
                                                   interpolation=cv2.INTER_LINEAR)
                mask_out[i, :, :, :] = cv2.resize(mask_cropped[i, :, :, :], (img_size, img_size),
                                                  interpolation=cv2.INTER_LINEAR)

        mask_out[np.nonzero(mask_out)] = 1.
        assert(np.setdiff1d(mask_out, [0., 1.]).size == 0)

        if flag == True:
            target_fn = np.transpose(target_fn, [0, 3, 1, 2])
            mask_out = np.transpose(mask_out, [0, 3, 1, 2])


        return target_fn, mask_out

    def get_train_alpha(self, zs_batch, trainEmbed=False):
        ''' get an alpha for training, return in format
            alpha_val_for_graph, alpha_val_for_get_target_np'''
        if trainEmbed == True:
            index_ = np.random.choice(self.num_panel)
            # Discrete
            alpha_val = self.embedding_alpha[index_]
            batch_size = zs_batch.shape[0]
            alpha_val_for_graph = np.ones((batch_size, self.Nsliders)) \
                                  * np.log(alpha_val)
            alpha_val_for_target = alpha_val
            # Can be negtive
            # print('Train alpha in Zoom:', np.log(alpha_val))

            return alpha_val_for_graph, alpha_val_for_target, index_

        #if self.walk_type == 'linear':
        coin = np.random.uniform(0, 1)
        # if coin <= 0.5:
        #     alpha_val = np.random.uniform(0.25, 1.)
        # else:
        #     alpha_val = np.random.uniform(1., 4.)

        if coin <= 0.5:
            alpha_val = np.random.uniform(0.5, 1.)
        else:
            alpha_val = np.random.uniform(1., 2.)

        batch_size = zs_batch.shape[0]
        alpha_val_for_graph = np.ones((batch_size, self.Nsliders)) \
                * np.log(alpha_val)
        alpha_val_for_target = alpha_val
        return alpha_val_for_graph, alpha_val_for_target, None

        # elif self.walk_type == 'NNz':
        #     alpha_val = np.random.randint(-self.N_f, self.N_f+1)
        #     alpha_val_for_graph = alpha_val + self.N_f
        #     alpha_val_for_target = self.eps ** alpha_val
        #     return alpha_val_for_graph, alpha_val_for_target

    def scale_test_alpha_for_graph(self, alpha, zs_batch, **kwargs):
        ''' map a scalar alpha to the appropriate shape,
            and do the desired transformation '''
        #if self.walk_type == 'linear':
        ####
        # alpha_val_for_graph = np.ones((batch_size, self.Nsliders)) * np.log(alpha_val)
        # alpha_val_for_target = alpha_val
        ###
        batch_size = zs_batch.shape[0]
        slider = np.log(alpha) * np.ones((batch_size, self.Nsliders))
        return slider

        # elif self.walk_type == 'NNz':
        #     return int(np.round(np.log(alpha) / np.log(self.eps)))

    def test_alphas(self):
        if self.walk_type == 'linear':
            return np.array([0.0625, 0.083, 0.125, 0.25, 0.5, 0.8, 1,
                             1.2, 2, 4, 8, 12, 16])
        elif self.walk_type == 'NNz':
            return np.array([0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16])
            # powers = np.arange(-2*self.N_f, 2*self.N_f + 1)
            # alphas = float(self.eps) ** powers
            # return alphas

    def vis_alphas(self, num_panels):
        # alpha range for visualization
        if self.walk_type == 'linear':
            alp = np.linspace(1, 8, num_panels//2 + 1)
            b = 1/alp
            alphas = np.concatenate((np.delete(b[::-1], -1), alp), axis=0)
            alphas = alphas[::-1]

        elif self.walk_type == 'NNz':
            powers = np.linspace(-1.5*self.N_f, 1.5*self.N_f, num_panels)
            alphas = float(self.eps) ** powers[::-1]
        return alphas

class ShiftTransform():
    def __init__(self):
        #self.alpha_max = self.img_size // 4

        self.alpha_original = 0
        self.max_x = self.img_size
        # NEW
        # Celeba
        # self.embed_alpha_max = 28
        # MNIST
        self.embed_alpha_max = 7
        self.alpha_max = 8
        # print('self.alpha_max: ',self.alpha_max)

        self.num_panel = 15
        self.embedding_alpha = np.linspace(-self.embed_alpha_max, self.embed_alpha_max, 15)

    def get_target_np(self, outputs_zs, alpha):
        raise NotImplementedError('Use ShiftXTransform or ShiftYTransform')

    def get_train_alpha(self, zs_batch, trainEmbed=False):
        ''' get an alpha for training, return in format
            alpha_val_wafor_graph, alpha_val_for_get_target_np'''
        if trainEmbed == True:
            index_ = np.random.choice(self.num_panel)

            alpha_val = self.embedding_alpha[index_]
            alpha_scaled = alpha_val / self.embed_alpha_max
            batch_size = zs_batch.shape[0]
            slider = np.ones((batch_size, self.Nsliders)) * alpha_scaled
            return slider, alpha_val, index_

        #if self.walk_type == 'linear':
        alpha_val = np.random.randint(-self.alpha_max, self.alpha_max)

        # alpha_val = np.random.randint(1, self.alpha_max)
        # coin = np.random.uniform(0, 1)
        # if coin <= 0.5:
        #     alpha_val = -alpha_val
        # print('alpha_val: ', self.alpha_max,  alpha_val)

        alpha_scaled = alpha_val / self.alpha_max
        batch_size = zs_batch.shape[0]
        slider = np.ones((batch_size, self.Nsliders)) * alpha_scaled
        return slider, alpha_val, None

        # elif self.walk_type == 'NNz':
        #     alpha_val = np.random.randint(-self.N_f, self.N_f+1)
        #     return alpha_val+self.N_f, alpha_val * self.eps, None

    def scale_test_alpha_for_graph(self, alpha, zs_batch, **kwargs):
        ''' map a scalar alpha to the appropriate shape,
            and do the desired transformation '''
        # if self.walk_type == 'linear':
        alpha_scaled = alpha / self.alpha_max
        batch_size = zs_batch.shape[0]
        slider = alpha_scaled * np.ones((batch_size, self.Nsliders))
        return slider

        # elif self.walk_type == 'NNz':
        #     # N_f = self.N_f
        #     # return one_hot_if_needed(alpha//self.eps + N_f, N_f*2+1)
        #     return int(np.round(alpha / self.eps))

    def test_alphas(self):
        return np.array([-200, -150, -100, -50, 0, 50, 100, 150, 200])
        #return np.array([-400, -300, -200, -150, -100, -50, 0, 50, 100, 150,
        #                 200, 300, 400])

    def vis_alphas(self, num_panels):
        # alpha range for visualization
        # if self.walk_type == 'linear':
        alphas = np.linspace(-400, 400, num_panels)
        # alphas = np.linspace(-150, 150, 7)
        # elif self.walk_type == 'NNz':
        #     alphas = np.linspace(-200, 200, num_panels)
        return alphas

class ShiftXTransform(ShiftTransform):
    def get_target_np(self, outputs_zs, alpha):
        flag = False
        if outputs_zs.shape[1] == 3 or outputs_zs.shape[1] == 1:
            flag = True
            outputs_zs = np.transpose(outputs_zs, [0, 2, 3, 1])

        img_size = outputs_zs.shape[1]
        mask_fn = np.ones(outputs_zs.shape)
        if alpha == 0:
            if flag == True:
                outputs_zs = np.transpose(outputs_zs, [0, 3, 1, 2])
                mask_fn = np.transpose(mask_fn, [0, 3, 1, 2])
            return outputs_zs, mask_fn

        M = np.float32([[1, 0, alpha], [0, 1, 0]])
        target_fn = np.zeros(outputs_zs.shape)
        mask_out = np.zeros(outputs_zs.shape)

        if outputs_zs.shape[-1] == 1:
            # grayscale
            for i in range(outputs_zs.shape[0]):
                target_fn[i, :, :, 0] = cv2.warpAffine(outputs_zs[i, :, :, 0], M, (img_size, img_size))
                mask_out[i, :, :, 0] = cv2.warpAffine(mask_fn[i, :, :, 0], M, (img_size, img_size))
        else:
            # rgb
            for i in range(outputs_zs.shape[0]):
                target_fn[i, :, :, :] = cv2.warpAffine(outputs_zs[i, :, :, :], M, (img_size, img_size))
                mask_out[i, :, :, :] = cv2.warpAffine(mask_fn[i, :, :, :], M, (img_size, img_size))
        mask_out[np.nonzero(mask_out)] = 1.
        assert (np.setdiff1d(mask_out, [0., 1.]).size == 0)

        if flag == True:
            target_fn = np.transpose(target_fn, [0, 3, 1, 2])
            mask_out = np.transpose(mask_out, [0, 3, 1, 2])

        return target_fn, mask_out

class ShiftYTransform(ShiftTransform):
    def get_target_np(self, outputs_zs, alpha):
        flag = False
        if outputs_zs.shape[1] == 3 or outputs_zs.shape[1] == 1:
            flag = True
            outputs_zs = np.transpose(outputs_zs, [0, 2, 3, 1])

        img_size = outputs_zs.shape[1]
        mask_fn = np.ones(outputs_zs.shape)
        if alpha == 0:
            if flag == True:
                outputs_zs = np.transpose(outputs_zs, [0, 3, 1, 2])
                mask_fn = np.transpose(mask_fn, [0, 3, 1, 2])
            return outputs_zs, mask_fn

        M = np.float32([[1,0,0],[0,1,alpha]])
        target_fn = np.zeros(outputs_zs.shape)
        mask_out = np.zeros(outputs_zs.shape)

        if outputs_zs.shape[-1] == 1:
            # grayscale
            for i in range(outputs_zs.shape[0]):
                target_fn[i, :, :, 0] = cv2.warpAffine(outputs_zs[i, :, :, 0], M, (img_size, img_size))
                mask_out[i, :, :, 0] = cv2.warpAffine(mask_fn[i, :, :, 0], M, (img_size, img_size))
        else:
            # rgb
            for i in range(outputs_zs.shape[0]):
                target_fn[i, :, :, :] = cv2.warpAffine(outputs_zs[i, :, :, :], M, (img_size, img_size))
                mask_out[i, :, :, :] = cv2.warpAffine(mask_fn[i, :, :, :], M, (img_size, img_size))
        mask_out[np.nonzero(mask_out)] = 1.
        assert (np.setdiff1d(mask_out, [0., 1.]).size == 0)

        if flag == True:
            target_fn = np.transpose(target_fn, [0, 3, 1, 2])
            mask_out = np.transpose(mask_out, [0, 3, 1, 2])

        return target_fn, mask_out

class Rotate2DTransform():

    def __init__(self):

        # CelebA
        # self.embed_alpha_max = 14

        # MNIST
        self.embed_alpha_max = 45
        self.alpha_max = 45
        self.num_panel = 15
        self.embedding_alpha = np.linspace(-self.embed_alpha_max, self.embed_alpha_max, self.num_panel)

    def get_target_np(self, outputs_zs, alpha):
        flag = False
        if outputs_zs.shape[1] == 3 or outputs_zs.shape[1] == 1:
            flag = True
            outputs_zs = np.transpose(outputs_zs, [0, 2, 3, 1])

        img_size = outputs_zs.shape[1]
        mask_fn = np.ones(outputs_zs.shape)
        if alpha == 0:
            if flag == True:
                outputs_zs = np.transpose(outputs_zs, [0, 3, 1, 2])
                mask_fn = np.transpose(mask_fn, [0, 3, 1, 2])
            return outputs_zs, mask_fn

        degree = alpha

        M = cv2.getRotationMatrix2D((img_size//2, img_size//2), degree, 1)
        target_fn = np.zeros(outputs_zs.shape)
        mask_out = np.zeros(outputs_zs.shape)

        ######
        if outputs_zs.shape[-1] == 1:
            # grayscale
            for i in range(outputs_zs.shape[0]):
                target_fn[i, :, :, 0] = cv2.warpAffine(outputs_zs[i, :, :, 0], M, (img_size, img_size))
                mask_out[i, :, :, 0] = cv2.warpAffine(mask_fn[i, :, :, 0], M, (img_size, img_size))
        else:
            # rgb
            for i in range(outputs_zs.shape[0]):
                target_fn[i, :, :, :] = cv2.warpAffine(outputs_zs[i, :, :, :], M, (img_size, img_size))
                mask_out[i, :, :, :] = cv2.warpAffine(mask_fn[i, :, :, :], M, (img_size, img_size))
        mask_out[np.nonzero(mask_out)] = 1.
        assert (np.setdiff1d(mask_out, [0., 1.]).size == 0)

        if flag == True:
            target_fn = np.transpose(target_fn, [0, 3, 1, 2])
            mask_out = np.transpose(mask_out, [0, 3, 1, 2])

        return target_fn, mask_out

    def get_train_alpha(self, zs_batch, trainEmbed=False):
        ''' get an alpha for training, return in format
            alpha_val_for_graph, alpha_val_for_get_target_np'''

        if trainEmbed == True:
            index_ = np.random.choice(self.num_panel)

            alpha_val = self.embedding_alpha[index_]
            alpha_scaled = alpha_val / self.embed_alpha_max
            batch_size = zs_batch.shape[0]
            slider = np.ones((batch_size, self.Nsliders)) * alpha_scaled
            return slider, alpha_val, index_

        # if self.walk_type == 'linear':
        alpha_val = np.random.randint(-self.alpha_max, self.alpha_max) # changed to increase this range

        # coin = np.random.uniform(0, 1)
        # if coin <= 0.5:
        #     alpha_val = -alpha_val

        alpha_scaled = alpha_val / self.alpha_max
        batch_size = zs_batch.shape[0]
        slider = np.ones((batch_size, self.Nsliders)) * alpha_scaled
        return slider, alpha_val, None

        # elif self.walk_type == 'NNz':
        #     alpha_val = np.random.randint(-self.N_f, self.N_f + 1)
        #     return alpha_val + self.N_f, alpha_val * self.eps

    def scale_test_alpha_for_graph(self, alpha, zs_batch, **kwargs):
        ''' map a scalar alpha to the appropriate shape,
            and do the desired transformation '''
        #if self.walk_type == 'linear':
        alpha_scaled = alpha / self.alpha_max
        batch_size = zs_batch.shape[0]
        slider = alpha_scaled * np.ones((batch_size, self.Nsliders))
        return slider

        # elif self.walk_type == 'NNz':
        #     # N_f = self.N_f
        #     # return one_hot_if_needed(alpha // self.eps + N_f, N_f * 2 + 1)
        #     return int(np.round(alpha / self.eps))

    def test_alphas(self):
        return np.linspace(-90, 90, 9)

    def vis_alphas(self, num_panels):
        # alpha range for visualization
        # if self.walk_type == 'linear':
        alphas = np.linspace(-90, 90, num_panels)
        # elif self.walk_type == 'NNz':
        #     # can expand range here beyond N_f * eps
        #     alphas = np.arange(-self.N_f * self.eps, self.N_f * self.eps + 1, self.eps)

        return alphas

class Rotate3DTransform():

    def __init__(self):
        self.alpha_max = 45

    def get_target_np(self, outputs_zs, alpha):
        mask_fn = np.ones(outputs_zs.shape)

        if alpha == 0:
            return outputs_zs, mask_fn

        target_fn = np.zeros(outputs_zs.shape)
        mask_out = np.zeros(outputs_zs.shape)
        for i in range(outputs_zs.shape[0]):
            it = ImageTransformer(outputs_zs[i,:,:,:], shape=None)
            target_fn[i,:,:,:] = it.rotate_along_axis(phi = alpha, dx = 0)
            it = ImageTransformer(mask_fn[i,:,:,:], shape=None)
            mask_out[i,:,:,:] = it.rotate_along_axis(phi = alpha, dx = 0)

        mask_out[np.nonzero(mask_out)] = 1.
        assert(np.setdiff1d(mask_out, [0., 1.]).size == 0)
        return target_fn, mask_out


    def get_train_alpha(self, zs_batch):
        ''' get an alpha for training, return in format
            alpha_val_for_graph, alpha_val_for_get_target_np'''
        # if self.walk_type == 'linear':
        alpha_val = np.random.randint(1, self.alpha_max)
        coin = np.random.uniform(0, 1)
        if coin <= 0.5:
            alpha_val = -alpha_val
        alpha_scaled = alpha_val / self.alpha_max
        batch_size = zs_batch.shape[0]
        slider = np.ones((batch_size, self.Nsliders)) * alpha_scaled
        return slider, alpha_val
        # elif self.walk_type == 'NNz':
        #     alpha_val = np.random.randint(-self.N_f, self.N_f + 1)
        #     return alpha_val + self.N_f, alpha_val * self.eps

    def scale_test_alpha_for_graph(self, alpha, zs_batch, **kwargs):
        ''' map a scalar alpha to the appropriate shape,
            and do the desired transformation '''
        #if self.walk_type == 'linear':
        alpha_scaled = alpha / self.alpha_max
        batch_size = zs_batch.shape[0]
        slider = alpha_scaled * np.ones((batch_size, self.Nsliders))
        return slider
        # elif self.walk_type == 'NNz':
        #     # N_f = self.N_f
        #     # return one_hot_if_needed(alpha // self.eps + N_f, N_f * 2 + 1)
        #     return int(np.round(alpha / self.eps))

    def test_alphas(self):
        # rot3d needs a large range to have an effect
        return np.linspace(-720, 720, 9)

    def vis_alphas(self, num_panels):
        # 3d rotate needs a large range to have an effect
        if self.walk_type == 'linear':
            alphas = np.linspace(-720, 720, num_panels)
        elif self.walk_type == 'NNz':
            # can expand range here beyond N_f * eps
            # alphas = np.arange(-self.N_f * self.eps, self.N_f * self.eps + 1, self.eps)
            alphas = np.linspace(-270, 270, num_panels)
        return alphas

def lerp(A, B, num_interps):
    alphas = np.linspace(-1.5, 2.5, num_interps)
    if A.shape != B.shape:
        raise ValueError('A and B must have the same shape to interpolate.')
    return np.array([(1-a)*A + a*B for a in alphas])

def slerp(A, B, num_interps):
    alphas = np.linspace(-1.5, 2.5, num_interps)
    if A.shape != B.shape:
        raise ValueError('A and B must have the same shape to interpolate.')
    omega = np.zeros((A.shape[0],1))
    for i in range(A.shape[0]):
        #angle = atan2(vector2.y, vector2.x) - atan2(vector1.y, vector1.x)
        tmp = np.dot(A[i],B[i])/(np.linalg.norm(A[i])*np.linalg.norm(B[i]))
        omega[i] = np.arccos(np.clip(tmp,0.0,1.0))+1e-9
        return np.array([(np.sin((1-a)*omega)/np.sin(omega))*A
                         + (np.sin(a*omega)/np.sin(omega))*B for a in alphas])

class FaceTransform():
    def __init__(self, atrr_name='Black_Hair'):
            self.atrr_name = atrr_name
            self.alpha_original = 1
            ### NEW
            self.num_panel = 6
            self.embed_alpha_max = 1
            self.embedding_alpha = np.linspace(0.0, 1.0, self.num_panel)

            # self.embedding_alpha = np.exp(np.linspace(0, 1, self.num_panel))
            #self.gt_model = relgan()
            self.alpha_max = 1

    def get_train_alpha(self, zs_batch, N_attr=40, trainEmbed=False):
        ''' get an alpha for training, return in format
            alpha_val_for_graph, alpha_val_for_get_target_np'''
        if trainEmbed == True:
            index_ = np.random.choice(self.num_panel)
            alpha_val = self.embedding_alpha[index_]
            alpha_scaled = alpha_val / self.embed_alpha_max
            batch_size = zs_batch.shape[0]
            slider = np.ones((batch_size, self.Nsliders)) * alpha_scaled
            return slider, alpha_val, index_


        alpha_val = np.random.uniform(0, 1, N_attr)
        alpha_scaled = alpha_val
        batch_size = zs_batch.shape[0]
        slider = np.ones((batch_size, self.Nsliders)) * alpha_scaled
        return slider, alpha_val, None



    def scale_test_alpha_for_graph(self, alpha, zs_batch, **kwargs):
        ''' map a scalar alpha to the appropriate shape,
            and do the desired transformation '''
        # if self.walk_type == 'linear':
        alpha_scaled = alpha # / self.alpha_max
        batch_size = zs_batch.shape[0]
        slider = alpha_scaled * np.ones((batch_size, self.Nsliders))
        return slider

        # elif self.walk_type == 'NNz':
        #     raise("ERROR")

    def test_alphas(self):
        if self.walk_type == 'linear':
            return np.linspace(0, 1, 10)
        else:
            raise('ERROR')

    def vis_alphas(self, num_panels):
        # alpha range for visualization
        if self.walk_type == 'linear':
            alphas = np.linspace(0, 1, num_panels)
        else:
            raise ('ERROR')
        return alphas


class SceneTransform():
    def __init__(self):
            self.alpha_max = 1

            # Non-linear embed
            self.num_panel = 6
            self.embed_alpha_max = 1
            self.embedding_alpha = np.linspace(0.0, 1.0, self.num_panel)

    def get_train_alpha(self, zs_batch, N_attr=40, trainEmbed=False):
        ''' get an alpha for training, return in format
            alpha_val_for_graph, alpha_val_for_get_target_np'''
        if trainEmbed == True:
            index_ = np.random.choice(self.num_panel)
            alpha_val = self.embedding_alpha[index_]

            batch_size = zs_batch.shape[0]
            slider = np.ones((batch_size, self.Nsliders)) * alpha_val
            return slider, alpha_val, index_

        # alpha_val = np.random.uniform(0, 1)
        alpha_val = np.random.uniform(-1, 1, N_attr) # 40
        batch_size = zs_batch.shape[0]
        slider = np.ones((batch_size, N_attr)) * alpha_val
        return slider, alpha_val, None


    def scale_test_alpha_for_graph(self, alpha, zs_batch, **kwargs):
        ''' map a scalar alpha to the appropriate shape,
            and do the desired transformation '''

        alpha_scaled = alpha
        batch_size = zs_batch.shape[0]
        slider = alpha_scaled * np.ones((batch_size, self.Nsliders))
        return slider


    def test_alphas(self):
        return np.linspace(0, 1, 10)

    def vis_alphas(self, num_panels):
        return np.linspace(0, 1, num_panels)


class ChairTransform():
    def __init__(self):
            self.alpha_max = 1
            # Non-linear embed
            self.num_panel = 6
            self.embed_alpha_max = 1
            self.embedding_alpha = np.linspace(0.0, 1.0, self.num_panel)

    def get_train_alpha(self, zs_batch, N_attr=2, trainEmbed=False):
        ''' get an alpha for training, return in format
            alpha_val_for_graph, alpha_val_for_get_target_np'''

        if trainEmbed == True:
            index_ = np.random.choice(self.num_panel)
            alpha_val = self.embedding_alpha[index_]

            batch_size = zs_batch.shape[0]
            slider = np.ones((batch_size, self.Nsliders)) * alpha_val
            return slider, alpha_val, index_



        # alpha_val = np.random.uniform(0, 1)
        alpha_val = np.random.uniform(-1, 1, N_attr) # 40
        batch_size = zs_batch.shape[0]
        slider = np.ones((batch_size, N_attr)) * alpha_val
        return slider, alpha_val, None



    def scale_test_alpha_for_graph(self, alpha, zs_batch, **kwargs):
        ''' map a scalar alpha to the appropriate shape,
            and do the desired transformation '''

        alpha_scaled = alpha
        batch_size = zs_batch.shape[0]
        slider = alpha_scaled * np.ones((batch_size, self.Nsliders))
        return slider


    def test_alphas(self):
        return np.linspace(0, 1, 10)

    def vis_alphas(self, num_panels):
        return np.linspace(0, 1, num_panels)


class dspritesTransform():
    def __init__(self):
            self.alpha_max = 1
            # Non-linear embed
            self.num_panel = 6
            self.embed_alpha_max = 1
            self.embedding_alpha = np.linspace(0.0, 1.0, self.num_panel)

            self.scale = [0.5, 0.6, 0.7, 0.8, 0.9, 1. ]

    def get_train_alpha(self, zs_batch, attrList=None, trainEmbed=False):
        ''' get an alpha for training, return in format
            alpha_val_for_graph, alpha_val_for_get_target_np'''

        # if trainEmbed == True:
        #     index_ = np.random.choice(self.num_panel)
        #     alpha_val = self.embedding_alpha[index_]
        #
        #     batch_size = zs_batch.shape[0]
        #     slider = np.ones((batch_size, self.Nsliders)) * alpha_val
        #     return slider, alpha_val, index_

        # alpha_val = np.random.uniform(0, 1)
        # TODO attrList
        N_attr = len(attrList)
        batch_size = zs_batch.shape[0]
        alpha_vals = []
        sliders = []
        if 'scale' in attrList:
            idx = np.random.choice(len(self.scale))
            alpha_val = self.scale[idx]
            slider = np.ones((batch_size, 1)) * alpha_val
            alpha_vals.append(alpha_val)
            sliders.append(slider)

        if 'x' in attrList:
            for _ in range(2):
                alpha_val = np.random.uniform(-1, 1)
                slider = np.ones((batch_size, 1)) * alpha_val
                alpha_vals.append(alpha_val)
                sliders.append(slider)

        if 'posx' in attrList:
            alpha_val = np.random.uniform(0, 1)
            slider = np.ones((batch_size, 1)) * alpha_val
            alpha_vals.append(alpha_val)
            sliders.append(slider)

        if 'posy' in attrList:
            alpha_val = np.random.uniform(0, 1)
            slider = np.ones((batch_size, 1)) * alpha_val
            alpha_vals.append(alpha_val)
            sliders.append(slider)

        alpha_val_new = np.hstack(alpha_vals)
        slider_new = np.hstack(sliders)

        # return slider, alpha_val, None
        return slider_new, alpha_val_new, None



    def scale_test_alpha_for_graph(self, alpha, zs_batch, **kwargs):
        ''' map a scalar alpha to the appropriate shape,
            and do the desired transformation '''

        alpha_scaled = alpha
        batch_size = zs_batch.shape[0]
        slider = alpha_scaled * np.ones((batch_size, self.Nsliders))
        return slider


    def test_alphas(self):
        return np.linspace(0, 1, 10)

    def vis_alphas(self, num_panels):
        return np.linspace(0, 1, num_panels)


class XrayTransform():
    def __init__(self, atrr_name=None):
        self.atrr_name = atrr_name
        self.alpha_original = 1
        ### NEW
        self.num_panel = 6
        self.embed_alpha_max = 1
        self.embedding_alpha = np.linspace(0.0, 1.0, self.num_panel)
        self.alpha_max = 1

    def get_train_alpha(self, zs_batch, N_attr=40, trainEmbed=False):
        ''' get an alpha for training, return in format
            alpha_val_for_graph, alpha_val_for_get_target_np'''
        if trainEmbed == True:
            """
            If only train with N alphas instead of a continuous alpha.
            This is an unused setting.
            """
            index_ = np.random.choice(self.num_panel)
            alpha_val = self.embedding_alpha[index_]
            alpha_scaled = alpha_val / self.embed_alpha_max
            batch_size = zs_batch.shape[0]
            slider = np.ones((batch_size, self.Nsliders)) * alpha_scaled
            return slider, alpha_val, index_

        alpha_val = np.random.uniform(0, 1, N_attr)
        alpha_scaled = alpha_val
        batch_size = zs_batch.shape[0]
        slider = np.ones((batch_size, self.Nsliders)) * alpha_scaled
        return slider, alpha_val, None



    def scale_test_alpha_for_graph(self, alpha, zs_batch, **kwargs):
        ''' map a scalar alpha to the appropriate shape,
            and do the desired transformation '''
        alpha_scaled = alpha
        batch_size = zs_batch.shape[0]
        slider = alpha_scaled * np.ones((batch_size, self.Nsliders))
        return slider

        # elif self.walk_type == 'NNz':
        #     raise("ERROR")

    def test_alphas(self):
        if self.walk_type == 'linear':
            return np.linspace(0, 1, 10)
        else:
            raise('ERROR')

    def vis_alphas(self, num_panels):
        # alpha range for visualization
        if self.walk_type == 'linear':
            alphas = np.linspace(0, 1, num_panels)
        else:
            raise ('ERROR')
        return alphas



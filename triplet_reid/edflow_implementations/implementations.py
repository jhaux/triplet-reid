import tensorflow as tf
import os
from importlib import import_module
import numpy as np

from scipy.misc import imresize

from edflow.iterators.model_iterator import HookedModelIterator
from edflow.hooks.hook import Hook
from edflow.iterators.resize import resize_float32

# path to checkpoint-25000 contained in
# https://github.com/VisualComputingInstitute/triplet-reid/releases/download/250eb1/market1501_weights.zip
TRIP_CHECK = os.environ.get('REID_CHECKPOINT',
                            '/home/johannes/Documents/Uni HD/Dr_J/'
                            'Projects/triplet_reid/triplet_reid/'
                            'pretrained/checkpoint-25000')

TRIP_W = 128
TRIP_H = 256


class reIdModel(object):
    '''ReID model as from the paper "In defense of the triplet loss"'''

    @property
    def name(self):
        return 'triplet_reid'

    def __init__(self,
                 model_name='resnet_v1_50',
                 head_name='fc1024',
                 w=TRIP_W,
                 h=TRIP_H,
                 nc=3,
                 edim=128,
                 is_train=False):
        '''Args:
            model_name (str): Which base model to use.
            head_name (str)L Which output head to use.
            w (int): Image width.
            h (int): Image height.
            nc (int): Image channels.A
            edim (int): Embedding dimension.
            is_train (bool): Is it training or not?
        '''

        model = import_module('triplet_reid.nets.' + model_name)
        head = import_module('triplet_reid.heads.' + head_name)

        self.images = im = tf.placeholder(tf.float32, shape=[None, h, w, nc])

        self.idn = 'my_triplet_is_the_best_triplet'

        with tf.variable_scope(self.idn):
            endpoints, body_prefix = model.endpoints(im,
                                                     is_training=is_train,
                                                     prefix=self.idn + '/')
            with tf.name_scope('head'):
                endpoints = head.head(endpoints,
                                      edim,
                                      is_training=is_train)

        self.embeddings = endpoints

        globs = tf.global_variables()
        self.variables = [v for v in globs if self.idn in v.name]

    @property
    def inputs(self):
        return {'image': self.images}

    @property
    def outputs(self):
        return {'embeddings': self.embeddings}


class reIdEvaluator(HookedModelIterator):
    def __init__(self, checkpoint, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.checkpoint = checkpoint

    def step_ops(self):
        return self.outputs

    def fit(self, *args, **kwargs):
        return self.iterate(*args, **kwargs)

    def initialize(self):
        pass

    def get_init_op(self):
        '''This overwrites the standart random initialization and forces
        the network to be initialized from the downloadable checkpoint.
        '''

        initialize_model(self.model, self.checkpoint, self.session)

        return tf.no_op()


def initialize_model(model, checkpoint, session=None):
    '''Loads weights from a checkpointfile and initializes the model.
    =======> THIS FUNCTION IS ONLY INTENDED TO BE USED HERE <=======
    '''

    sess = session or tf.Session()

    sess.run(tf.global_variables_initializer())

    if checkpoint is None:
        raise ValueError('The reIdEvaluator needs a checkpoint from which '
                         'to initialize the model.')

    var_map = {}
    for v in model.variables:
        vn = v.name.strip(model.idn).strip('/').strip(':0')
        var_map = {vn: v}

    tf.train.Saver(var_map).restore(sess, checkpoint)


class PrepData(Hook):
    def __init__(self, image_pl, w=TRIP_W, h=TRIP_H):
        '''Rescales the images fed to the reid model.'''
        self.impl = image_pl
        self.w = w
        self.h = h

    def before_step(self, step, fetches, feeds, batch):
        image = batch['image']

        feeds[self.impl] = resize(image, self.w, self.h)


def resize(image, w=TRIP_W, h=TRIP_H):
    ims = []
    for im in image:
        if w == h // 2 and im.shape[0] == im.shape[1]:
            im = resize_float32(im, h)
            im = im[:, h//4:h//4+w, :]
        else:
            assert False
            im = imresize(im, [w, h])
        ims += [im]
    ims = np.array(ims)
    # edflow works with images in [-1,1] but reid net expects [0,255]
    ims = (ims+1.0)*127.5

    return ims


def getReIdEvaluator(config,
                     root,
                     model,
                     hook_freq):

    hooks = [PrepData(model.inputs['image'])]

    checkpoint = TRIP_CHECK

    return reIdEvaluator(checkpoint,
                         model,
                         num_epochs=config['num_epochs'],
                         hooks=hooks,
                         hook_freq=hook_freq,
                         bar_position=0,
                         gpu_mem_growth=False,
                         gpu_mem_fraction=None,
                         nogpu=False)


def pretrainedReIdModel(config):
    return reIdModel()


class reIdMetricFn(object):
    def __init__(self, session=None, sess_config=None,
                 allow_growth=None, mem_frac=None, nogpu=False):
        sess_config = sess_config or tf.ConfigProto()

        if allow_growth is not None:
            sess_config.gpu_options.allow_growth = allow_growth
        if mem_frac is not None:
            sess_config.gpu_options.per_process_gpu_memory_fraction = mem_frac

        if nogpu:
            sess_config.device_count["GPU"] = 0

        self.session = session or tf.Session(config=sess_config)

        self.model = reIdModel()
        initialize_model(self.model, TRIP_CHECK, self.session)

    def get_embeddings(self, images):
        fetches = self.model.outputs
        feeds = {self.model.inputs['image']: resize(images)}

        return self.session.run(fetches, feed_dict=feeds)['embeddings']

    def __call__(self, image, generated):
        if image.shape[-1] == 4:
            image = image[..., :3]

        if generated.shape[-1] == 4:
            generated = generated[..., :3]

        orig = self.get_embeddings(image)
        genr = self.get_embeddings(generated)

        orig_e = orig['emb']
        genr_e = genr['emb']

        diff = orig_e - genr_e

        dist = np.linalg.norm(diff, axis=1)

        return dist

import tensorflow as tf
import os
from importlib import import_module
import numpy as np

from scipy.misc import imresize

from edflow.iterators.model_iterator import HookedModelIterator, TFHookedModelIterator
from edflow.hooks.hook import Hook
from edflow.hooks.train_hooks import LoggingHook, CheckpointHook
from edflow.hooks.util_hooks import IntervalHook
from edflow.iterators.resize import resize_float32
from edflow.project_manager import ProjectManager

import triplet_reid.loss as loss

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

        self.model_name = 'my_triplet_is_the_best_triplet'

        with tf.variable_scope(self.model_name):
            self.images = tf.placeholder(tf.float32, shape=[None, h, w, nc])
            endpoints, body_prefix = model.endpoints(self.images,
                                                     is_training=is_train,
                                                     prefix=self.model_name + '/')
            with tf.name_scope('head'):
                endpoints = head.head(endpoints,
                                      edim,
                                      is_training=is_train)

        self.embeddings = endpoints

        globs = tf.global_variables()
        self.variables = [v for v in globs if self.model_name in v.name]

    @property
    def inputs(self):
        return {'image': self.images}

    @property
    def outputs(self):
        return {'embeddings': self.embeddings}


class EdflowModel(reIdModel):
    def __init__(self, config):
        self.config = config
        base_config = dict(
                 model_name=config.get("backbone", 'resnet_v1_50'),
                 head_name=config.get("head_name", 'fc1024'),
                 w=config.get("input_width", TRIP_W),
                 h=config.get("input_height", TRIP_H),
                 nc=config.get("nc", 3),
                 edim=config.get("edim", 128),
                 is_train=not config.get("test_mode", False))
        super().__init__(**base_config)


class Unstack(Hook):
    def __init__(self, iterator):
        self.iterator = iterator

    def before_step(self, index, fetches, feeds, batch):
        # modify feeds to unstack examples
        image = feeds[self.iterator.model.inputs["image"]]
        bs, n_views, h, w, c = image.shape
        image = image.reshape(bs*n_views, h, w, c)
        # possibly remove alpha channel
        assert c in [3,4]
        if c == 4:
            image = image[...,:3]
        feeds[self.iterator.model.inputs["image"]] = image

        pids = batch["pid"]
        bs, n_views = pids.shape
        pids = pids.reshape(bs*n_views)
        feeds[self.iterator.pid_placeholder] = pids


class Trainer(TFHookedModelIterator):
    def __init__(self, config, root, model, **kwargs):
        unstackhook = Unstack(self)
        kwargs["hook_freq"] = 1
        kwargs["hooks"] = [unstackhook]
        super().__init__(config, root, model, **kwargs)
        self._init_step_ops()

    def initialize(self, checkpoint_path = None):
        # if none given use market pretrained
        assert checkpoint_path is None
        checkpoint_path = checkpoint_path or TRIP_CHECK
        initialize_model(self.model, checkpoint_path, self.session)

    def step_ops(self):
        return self._step_ops

    def _init_step_ops(self):
        # additional inputs
        self.pid_placeholder = tf.placeholder(tf.string, shape=[None])

        # loss
        endpoints = self.model.embeddings
        dists = loss.cdist(endpoints['emb'], endpoints['emb'],
                metric=self.config.get("metric", "euclidean"))
        losses, train_top1, prec_at_k, _, neg_dists, pos_dists = (
                loss.LOSS_CHOICES["batch_hard"](
                    dists, self.pid_placeholder, self.config.get("margin", "soft"),
                    batch_precision_at_k=self.config.get("n_views", 4)-1))

        # Count the number of active entries, and compute the total batch loss.
        loss_mean = tf.reduce_mean(losses)

        # train op
        learning_rate = self.config.get("learning_rate", 3e-4)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.minimize(loss_mean)
        self._step_ops = train_op

        loghook = LoggingHook(
                logs = {"loss": loss_mean},
                images = {"image": self.model.inputs["image"]},
                root_path = ProjectManager().train,
                interval = 1)
        ckpt_hook = CheckpointHook(
                root_path = ProjectManager().checkpoints,
                variables = tf.global_variables(),
                modelname = self.model.name,
                step = self.get_global_step,
                interval = 1,
                max_to_keep = 2)
        ihook = IntervalHook([loghook, ckpt_hook],
                interval = 1, modify_each = 1,
                max_interval = self.config.get("log_freq", 1000))
        self.hooks.append(ihook)


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
        vn = v.name.strip(model.model_name).strip('/').strip(':0')
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

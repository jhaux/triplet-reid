import tensorflow as tf
import os, json, pickle
from importlib import import_module
import numpy as np

from scipy.misc import imresize
from scipy.spatial.distance import cdist
from sklearn.metrics import average_precision_score

from edflow.iterators.model_iterator import HookedModelIterator, TFHookedModelIterator
from edflow.hooks.hook import Hook
from edflow.hooks.train_hooks import LoggingHook, CheckpointHook
from edflow.hooks.evaluation_hooks import (
        WaitForCheckpointHook, RestoreTFModelHook, KeepBestCheckpoints)
from edflow.hooks.util_hooks import IntervalHook
from edflow.iterators.resize import resize_float32
from edflow.project_manager import ProjectManager
from edflow.custom_logging import get_logger
from edflow.util import retrieve, walk

import triplet_reid.loss as loss
from triplet_reid.excluders.ntugems import Excluder

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
            # edflow works with images in [-1,1] but reid net expects [0,255]
            self.rescaled_images = (self.images+1.0)*127.5
            input_images = self.rescaled_images
            endpoints, body_prefix = model.endpoints(input_images,
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
        # (bs, h, w, c) in [-1, 1]
        return {'image': self.images}

    @property
    def outputs(self):
        # (bs, 128
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

        # add pids for training
        pids = batch["pid"]
        bs, n_views = pids.shape
        pids = pids.reshape(bs*n_views)
        feeds[self.iterator.pid_placeholder] = pids


class Trainer(TFHookedModelIterator):
    def __init__(self, config, root, model, **kwargs):
        unstackhook = Unstack(self)
        kwargs["hook_freq"] = 1
        kwargs["hooks"] = [unstackhook]
        super().__init__(config, root, model, num_epochs = config["num_epochs"], **kwargs)
        self._init_step_ops()
        restorer = RestoreTFModelHook(variables = tf.global_variables(),
                                      checkpoint_path = ProjectManager().checkpoints,
                                      global_step_setter = self.set_global_step)
        self.restorer = restorer

    def initialize(self, checkpoint_path = None):
        # if none given use market pretrained
        if checkpoint_path is None:
            checkpoint_path = TRIP_CHECK
            initialize_model(self.model, checkpoint_path, self.session)
        else:
            with self.session.as_default():
                self.restorer(checkpoint_path)

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
        self.logger.info("Training with learning rate: {}".format(learning_rate))
        optimizer = tf.train.AdamOptimizer(learning_rate)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.minimize(loss_mean)
        self._step_ops = train_op

        tolog = {"loss": loss_mean, "top1": train_top1,
                "prec@{}".format(self.config.get("n_views", 4)-1): prec_at_k}
        loghook = LoggingHook(
                logs = tolog,
                scalars = tolog,
                images = {"image": self.model.inputs["image"]},
                root_path = ProjectManager().train,
                interval = 1)
        ckpt_hook = CheckpointHook(
                root_path = ProjectManager().checkpoints,
                variables = tf.global_variables(),
                modelname = self.model.name,
                step = self.get_global_step,
                interval = self.config.get("ckpt_freq", 1000),
                max_to_keep = None)
        self.hooks.append(ckpt_hook)
        ihook = IntervalHook([loghook],
                interval = 1, modify_each = 1,
                max_interval = self.config.get("log_freq", 1000))
        self.hooks.append(ihook)


def initialize_model(model, checkpoint, session=None):
    '''Loads weights from a checkpointfile and initializes the model.
    This function is just for the case of restoring the market-1501 pretrained
    model because we have to map variable names correctly. For newly written
    checkpoints use the RestoreCheckpointHook.
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
        var_map[vn] = v

    tf.train.Saver(var_map).restore(sess, checkpoint)
    get_logger("initialize_model").info("Restored model from {}".format(checkpoint))


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

    return ims


class reIdMetricFn(object):
    """
    Implements a metric to compare two images based on the difference between
    their reid embeddings. Use the environment variable REID_CHECKPOINT to
    select which checkpoint is to be used for computing embeddings.
    """
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
        if os.path.basename(TRIP_CHECK) == "checkpoint-25000":
            # assume that this is Market pretrained checkpoint
            initialize_model(self.model, TRIP_CHECK, self.session)
        else:
            # assume this is checkpoint trained by ourselve
            with self.session.as_default():
                restorer = RestoreTFModelHook(variables = tf.global_variables(),
                                              checkpoint_path = None)
                restorer(TRIP_CHECK)


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


class EvalHook(Hook):
    def __init__(self, iterator):
        self.iterator = iterator
        self.logger = get_logger(self, "latest_eval")
        self.global_step = self.iterator.get_global_step
        self.root = ProjectManager().latest_eval
        self.tb_saver = tf.summary.FileWriter(self.root)

    def before_epoch(self, *args, **kwargs):
        self.data = dict()

    def before_step(self, step, fetches, feeds, batch):
        for key in ["pid", "name", "dataset_index_"]:
            if key in self.data:
                self.data[key] = np.concatenate([self.data[key], batch[key]])
            else:
                self.data[key] = batch[key]

    def after_step(self, step, results):
        for key in ["step_ops/emb"]:
            result = retrieve(key, results)
            if key in self.data:
                self.data[key] = np.concatenate([self.data[key], result])
            else:
                self.data[key] = result

    def after_epoch(self, ep):
        query_data = dict()
        gallery_data = dict()
        for key in self.data:
            query_data[key] = self.data[key][self.data["dataset_index_"] == 0]
            gallery_data[key] = self.data[key][self.data["dataset_index_"] == 1]

        # evaluation as in original evaluate.py
        distances = cdist(query_data["step_ops/emb"], gallery_data["step_ops/emb"],
                metric = "euclidean")

        # save everything required for evaluation
        eval_data = {
                "query_data": query_data,
                "gallery_data": gallery_data,
                "distances": distances}
        picklename = '{:0>6d}_eval_data.p'.format(self.global_step())
        picklename = os.path.join(self.root, picklename)
        with open(picklename, "wb") as f:
            pickle.dump(eval_data, f)

        # Compute the pid matches
        pid_matches = gallery_data["pid"][None] == query_data["pid"][:,None]

        # Get a mask indicating True for those gallery entries that should
        # be ignored for whatever reason (same camera, junk, ...) and
        # exclude those in a way that doesn't affect CMC and mAP.
        excluder = Excluder(gallery_data["name"])
        mask = excluder(query_data["name"])
        distances[mask] = np.inf
        pid_matches[mask] = False

        # Keep track of statistics. Invert distances to scores using any
        # arbitrary inversion, as long as it's monotonic and well-behaved,
        # it won't change anything.
        aps = []
        cmc = np.zeros(len(gallery_data["pid"]), dtype=np.int32)
        scores = 1 / (1 + distances)
        for i in range(len(distances)):
            ap = average_precision_score(pid_matches[i], scores[i])

            if np.isnan(ap):
                logwarn = self.logger.warn
                logwarn()
                logwarn("WARNING: encountered an AP of NaN!")
                logwarn("This usually means a person only appears once.")
                logwarn("In this case, it's because of {}.".format(query_data["fid"][i]))
                logwarn("I'm excluding this person from eval and carrying on.")
                logwarn()
                continue

            aps.append(ap)
            # Find the first true match and increment the cmc data from there on.
            k = np.where(pid_matches[i, np.argsort(distances[i])])[0][0]
            cmc[k:] += 1

        # Compute the actual cmc and mAP values
        cmc = cmc / len(query_data["pid"])
        mean_ap = np.mean(aps)

        # Save important data
        map_ = np.array([mean_ap, 0.0])
        metric = {"mAP": map_, "-mAP": -map_}
        name = '{:0>6d}_metrics'.format(self.global_step())
        name = os.path.join(self.root, name)
        np.savez_compressed(name, **metric)

        # for visualization in tb
        summary = tf.Summary()
        summary.value.add(tag="mAP", simple_value=map_[0])
        self.tb_saver.add_summary(summary, self.global_step())
        self.tb_saver.flush()

        # Print out a short summary.
        self.iterator.logger.info(
                'step: {:06} | mAP: {:.2%} | top-1: {:.2%} top-2: {:.2%} | top-5: {:.2%} | top-10: {:.2%}'.format(
                    self.global_step(), mean_ap, cmc[0], cmc[1], cmc[4], cmc[9]))



class Evaluator(TFHookedModelIterator):
    def __init__(self, config, root, model, **kwargs):
        config["eval_forever"] = True
        super().__init__(config, root, model, num_epochs = 1, **kwargs)

        restorer = RestoreTFModelHook(variables = tf.global_variables(),
                                      checkpoint_path = ProjectManager().checkpoints,
                                      global_step_setter = self.set_global_step)
        self.restorer = restorer
        def has_eval(checkpoint):
            global_step = restorer.parse_global_step(checkpoint)
            eval_file = os.path.join(ProjectManager().latest_eval,
                                     "{:06}_metrics.npz".format(global_step))
            return not os.path.exists(eval_file)

        waiter = WaitForCheckpointHook(checkpoint_root = ProjectManager().checkpoints,
                                       filter_cond = has_eval,
                                       callback = restorer)
        evaluation = EvalHook(self)

        manager = KeepBestCheckpoints(checkpoint_root = ProjectManager().checkpoints,
                                      metric_template = os.path.join(
                                          ProjectManager().latest_eval,
                                          "{:06}_metrics.npz"),
                                      metric_key = "-mAP",
                                      n_keep = 2)
        self.hooks += [
                waiter,
                evaluation]
        if config.get("manage_checkpoints", False):
                self.hooks += [manager]
        self.initialize()

    def step_ops(self):
        return {"emb": self.model.outputs["embeddings"]["emb"]}

    def initialize(self, checkpoint_path = None):
        # if none given use market pretrained
        if checkpoint_path is None:
            checkpoint_path = checkpoint_path or TRIP_CHECK
            initialize_model(self.model, checkpoint_path, self.session)
        else:
            self.restorer(checkpoint_path)

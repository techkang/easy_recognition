import multiprocessing

from yacs.config import CfgNode as CN

_C = CN()

_C.base_cfg = ()
_C.output_dir = "output"
_C.trainer = "Trainer"

# Benchmark different cudnn algorithms.
# If x instances have very different sizes, this option will have large overhead
# for about 10k iterations. It usually hurts num_classes time, but can benefit for certain models.
# If x instances have the same or similar sizes, benchmark is often helpful.
_C.cudnn_benchmark = True
_C.device = "cuda"
_C.eval_only = False
_C.test_only = False
_C.num_gpus = 1
_C.start_iter = -1
_C.random_seed = -1
_C.log_name = ""
_C.tqdm_disable = False
_C.result_path = ""

_C.model = CN()
_C.model.name = "Recognizer"
_C.model.preprocessor = ""
_C.model.backbone = "VeryDeepVgg"
_C.model.encoder = ""
_C.model.decoder = "CRNNDecoder"
_C.model.loss = "CTCLoss"
_C.model.convertor = "CTCConvertor"
_C.model.weights = ""

_C.dataset = CN()
_C.dataset.path = "dataset"
_C.dataset.name = "OCRDataset"
_C.dataset.gray_scale = False

_C.dataset.train = CN()
_C.dataset.train.dataset = ("syn90k",)
_C.dataset.train.repeat = (1, )
_C.dataset.train.height = 32
_C.dataset.train.min_width = 32
_C.dataset.train.max_width = 512
_C.dataset.train.divisible = 0
_C.dataset.train.pad_position = "uniform"
_C.dataset.train.blur = False
_C.dataset.train.loader = "lmdb_loader"

_C.dataset.eval = CN()
_C.dataset.eval.dataset = ("ic13", )
_C.dataset.eval.repeat = (1, )
_C.dataset.eval.height = 32
_C.dataset.eval.min_width = 32
_C.dataset.eval.max_width = 512
_C.dataset.eval.divisible = 0
_C.dataset.eval.pad_position = "right-bottom"
_C.dataset.eval.blur = False
_C.dataset.eval.loader = "caffe_loader"

# test dataset will not be used for now
_C.dataset.test = CN()

_C.charset = CN()
_C.charset.name = "AsciiCharset"
_C.charset.corpus = ""
_C.charset.target_length = 30
_C.charset.sep = ""
_C.charset.use_space = False

_C.dataloader = CN()
_C.dataloader.collector = "default"
_C.dataloader.num_workers = min(multiprocessing.cpu_count(), 10)
_C.dataloader.prefetch_factor = 2
_C.dataloader.batch_size = 128
_C.dataloader.eval_batch_size = 0
_C.dataloader.eval_num_workers = -1

_C.optimizer = CN()
_C.optimizer.name = "Adam"

_C.lr_scheduler = CN()
_C.lr_scheduler.name = "MultiStepLR"
_C.lr_scheduler.start_lr = 0.1
_C.lr_scheduler.base_lr = 1e-3
_C.lr_scheduler.end_lr = 0.01
_C.lr_scheduler.warm_up_end_iter = 0.05
_C.lr_scheduler.cosine_start_iter = 0.7

_C.solver = CN()
_C.solver.max_iter = 10000
_C.solver.save_interval = 5000
_C.solver.eval_interval = 500
_C.solver.amp = False

_C.tensorboard = CN()
_C.tensorboard.enable = True
_C.tensorboard.save_freq = 500
_C.tensorboard.image_num = 100
_C.tensorboard.name = ""
_C.tensorboard.failed_only = False

_C.crnn = CN()
_C.crnn.inner_channel = 512

_C.swin = CN()
_C.swin.config = 0

_C.abi = CN()
_C.abi.language_model_weights = ""
_C.abi.vision_model_weights = ""
_C.abi.iter_time = 3
_C.abi.align = "cross"
_C.abi.freeze_language = False

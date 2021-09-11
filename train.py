import os

import trainer
from config import default_argument_parser, init_logging, setup
from tools.launch import launch_job


def main(args, config):
    init_logging(args, config)

    main_trainer = getattr(trainer, config.trainer)(config, resume=args.resume)
    if config.eval_only:
        main_trainer.eval()
        return
    if config.test_only:
        main_trainer.test()
        return
    main_trainer.train()
    main_trainer.test()


if __name__ == "__main__":
    arg = default_argument_parser().parse_args()
    if arg.visible_gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = arg.visible_gpus
    cfg = setup(arg)
    launch_job(main, arg, cfg)

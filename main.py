from argparse import ArgumentParser

parser = ArgumentParser(description="DiffLightning: a Lightning implementation of Diffusion Models")
# -- TRAIN ARGS --

parser.add_argument("-t", "--train", action="store_true", help="Train diffusion models on specified config file")
parser.add_argument("--config", type=str, help="Path to config file for sampling or resuming training", default="./config/training_config.yaml")

# -- SAMPLE ARGS --
parser.add_argument("-s", "--sample", action="store_true", help="Sample from a trained model")
parser.add_argument("-n", "--n_samples", type=int, help="Number of samples to generate", default=16)
parser.add_argument("-p", "--save_path", type=str, help="Path to save samples", default=None)

## -- CHECKPOINT PATH FOR TRAINING AND SAMPLING --
parser.add_argument("--ckpt", type=str, help="Path to checkpoint", default=None)

args = parser.parse_args()

if args.train:
    from src.training import DiffusionModelTrainer
    trainer = DiffusionModelTrainer(args.config)
    trainer.train(args.ckpt)
    trainer.test()
elif args.sample:
    print("Sampling from loaded model")

    if args.ckpt is None:
        print("Checkpoint path is required for sampling")
        exit(1)

    from src.sampling import DiffusionModelSampler
    sampler = DiffusionModelSampler(args.ckpt)
    sampler.sample(n_samples=args.n_samples, save_path=args.save_path)
else:
    print("Invalid arguments: check -h or --help for more info")
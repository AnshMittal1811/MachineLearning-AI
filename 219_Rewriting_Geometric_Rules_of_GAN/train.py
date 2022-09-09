import trainers
from option.option import TrainOptions


def main():
    opt = TrainOptions().parse()
    trainer = trainers.create_trainer(opt)
    trainer.fit()


if __name__ == "__main__":
    main()

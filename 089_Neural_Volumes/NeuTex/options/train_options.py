from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        self.is_train = True

        parser.add_argument(
            "--print_freq",
            type=int,
            default=100,
            help="frequency of showing training results on console",
        )

        parser.add_argument(
            "--lr", type=float, default=0.001, help="initial learning rate"
        )
        parser.add_argument(
            "--lr_policy",
            type=str,
            default="lambda",
            help="learning rate policy: lambda|step|plateau",
        )
        parser.add_argument(
            "--lr_decay_iters",
            type=int,
            default=50,
            help="multiply by a gamma every lr_decay_iters iterations",
        )

        parser.add_argument(
            "--train_and_test",
            type=int,
            default=0,
            help="train and test at the same time",
        )
        parser.add_argument("--test_num", type=int, default=1, help="test num")
        parser.add_argument("--test_freq", type=int, default=500, help="test frequency")

        parser.add_argument(
            "--niter", type=int, default=100, help="# of iter at starting learning rate"
        )
        parser.add_argument(
            "--niter_decay",
            type=int,
            default=100,
            help="# of iter to linearly decay learning rate to zero",
        )

        parser.add_argument(
            "--save_iter_freq", type=int, default=1000, help="saving frequency"
        )

        parser.add_argument(
            "--load_subnetworks_dir",
            type=str,
            default=None,
            help="directory to load subnetworks from",
        )
        parser.add_argument(
            "--load_subnetworks",
            type=str,
            default="",
            help="a comma separated list of subnetworks to load",
        )
        parser.add_argument(
            "--load_subnetworks_epoch",
            type=str,
            default="latest",
            help="subnetwork epoch to load",
        )
        parser.add_argument(
            "--freeze_subnetworks",
            type=str,
            default=None,
            help="subnetworks to freeze before training",
        )

        return parser

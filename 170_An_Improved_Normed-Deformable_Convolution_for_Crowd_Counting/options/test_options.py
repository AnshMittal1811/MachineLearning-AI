
from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # TODO: implemented me
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.isTrain = False

        return parser
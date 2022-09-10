# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


class BatchCollator:
    def __call__(self, batch):
        transposed_batch = list(zip(*[sample for sample in batch if sample[0] is not None]))

        if len(transposed_batch) == 0:
            return "error", None
        else:
            img_ids = transposed_batch[0]
            targets = transposed_batch[1]
        return img_ids, targets

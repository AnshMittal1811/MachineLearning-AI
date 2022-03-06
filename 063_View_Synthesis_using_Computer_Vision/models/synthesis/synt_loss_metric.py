from models.synthesis.losses import *
from models.synthesis.metrics import *


class SynthesisLossRGBandSeg(nn.Module):
    """
    Class for simultaneous calculation of SynthesisLoss on RGB and SEG predictions.
    Can be used for pretraining the (rgb + seg) refinement network on only static cases.
    """
    def __init__(self,
                 rgb_losses=['1.0_l1', '10.0_content'],
                 seg_losses=['1.0_l1', '10.0_content']):
        super().__init__()

        self.rgb_loss = SynthesisLoss(rgb_losses)
        self.seg_loss = SynthesisLoss(seg_losses)

    def forward(self, pred_img, gt_img, pred_seg, gt_seg, input_mask=None, gt_output_mask=None):

        if input_mask is not None or gt_output_mask is not None:
            raise ValueError(f"You provided dynamic masks to the SynthesisLoss meant for static training. Either remove the dynamic masks from the call or use a loss that can handle dynamic masks.")

        rgb_results = self.rgb_loss(pred_img, gt_img)
        seg_results = self.seg_loss(pred_seg, gt_seg)

        # sum up rgb and seg results without additional weighting.
        # rgb and seg share the same keys as they come from the same class
        results = {key: rgb_results[key] + seg_results[key] for key in rgb_results.keys()}

        return results


class SceneEditingAndSynthesisLoss(nn.Module):
    """
    Class for simultaneous calculation of SynthesisLoss and SceneEditingLoss.

    It calculates the SynthesisLoss for the rgb image prediction in comparison with the gt rgb image.
    Since we do not have gt rgb image with movements, we only compare with the (static, non-moved) gt rgb image at the
    regions in the image that contain no movement (e.g. input and gt dynamics mask gets not compared).

    It calculates the SceneEditingLoss for the seg image prediction in comparison with the gt seg image.
    """
    def __init__(self,
                 synthesis_losses=['1.0_l1', '10.0_content'],
                 scene_editing_weight=1.0,
                 scene_editing_lpregion_params=[1.0, 1.0, 3.0],
                 scene_editing_ignore_input_mask_region=False):
        super().__init__()

        self.synthesis_loss = SynthesisLoss(synthesis_losses, True)
        self.scene_editing_loss = SceneEditingLoss(weight=scene_editing_weight,
                                                   lpregion_params=scene_editing_lpregion_params,
                                                   ignore_input_mask_region=scene_editing_ignore_input_mask_region)

    def forward(self, pred_img, gt_img, pred_seg, gt_seg, input_mask, gt_output_mask):
        synthesis_results = self.synthesis_loss(pred_img, gt_img, input_mask, gt_output_mask)
        scene_editing_results = self.scene_editing_loss(pred_seg, gt_seg, input_mask, gt_output_mask)

        results = {**synthesis_results, **scene_editing_results}
        results["Total Loss"] = synthesis_results["Total Loss"] + scene_editing_results["Total Loss"]

        return results


class SceneEditingLoss(nn.Module):
    """
    Calculates the LPRegionLoss over the segmentation prediction at the given movement masks.
    """

    def __init__(self, weight=1.0, lpregion_params=[1.0, 0.0, 2.0], ignore_input_mask_region=False):
        super().__init__()

        self.weight = weight
        self.region_lp = LPRegionLoss(*lpregion_params)
        self.ignore_input_mask_region = ignore_input_mask_region

        print("Loss {} with weight {} and params {} and ignore_input_mask_region {}".format(type(self.region_lp).__name__, self.weight, lpregion_params, ignore_input_mask_region))

        if torch.cuda.is_available():
            self.region_lp = self.region_lp.cuda()

    '''
    def calculate_predicted_mask(self, pred_seg, gt_seg, gt_output_mask):
        # TODO how to vectorize this?
        bs = gt_output_mask.shape[0]
        c = gt_seg.shape[1]
        color = torch.zeros(bs, c, 1, 1, device=gt_output_mask.get_device())
        for i in range(bs):
            # get the first position where mask is true (nonzero)
            # TODO support more than one color --> find all colors in gt_output_mask @ gt_img
            # TODO what if in gt image the color is not consistent as well e.g. due to downsampling + antialiasing???
            color_index = torch.nonzero(gt_output_mask[i].squeeze(), as_tuple=False)[0] # TODO DEBUG: CHANGE BACK TO [0]

            # get the color from gt_img at that position
            color[i] = gt_seg[i, :, color_index[0], color_index[1]].unsqueeze(1).unsqueeze(2)

        # find all places where the color is equal in pred_img (comparison is per channel here)
        # TODO add "nearest color" search if color is not exactly the same? Does this make sense. We could see it as punishment when color is not exactly similar?
        # But we could use gradients when we search for exact same color and it is not exactly the same
        # Also: Floating point precision???
        color_equal_per_channel = torch.eq(pred_seg, color) # TODO DEBUG: CHANGE BACK TO pred_img

        # mulitply it here to still have gradients back to pred_img
        pred_output_mask_per_channel = pred_seg * color_equal_per_channel

        # mulitply 3x boolean values. Will only be True, when all are True (is a differentiable way of doing bitwise_and)
        pred_output_mask = (pred_output_mask_per_channel[:, 0] * pred_output_mask_per_channel[:, 1] * pred_output_mask_per_channel[:, 2]).unsqueeze(1)

        for i in range(bs):
            import matplotlib.pyplot as plt
            plt.imshow(pred_img[i].permute((1,2,0)).cpu().detach().numpy())
            plt.show()

            plt.imshow(gt_img[i].permute((1, 2, 0)).cpu().detach().numpy())
            plt.show()

            plt.imshow(gt_output_mask[i].squeeze().cpu().detach().numpy())
            plt.show()

            plt.imshow(pred_output_mask[i].squeeze().cpu().detach().numpy().squeeze())
            plt.show()

        return pred_output_mask
    '''

    def forward(self, pred_seg, gt_seg, input_mask, gt_output_mask):
        # calculate predicted_mask
        # pred_output_mask = self.calculate_predicted_mask(pred_seg, gt_seg, gt_output_mask)

        # pass to lp region loss: higher_region is everything that was moved (input, gt, pred), lower_region is the rest (unmoved pixels)
        #merged_output_mask = (pred_output_mask > 0) | gt_output_mask | input_mask
        merged_output_mask = gt_output_mask | input_mask
        if self.ignore_input_mask_region:
            no_weight_mask = input_mask & ~gt_output_mask
            region_lp = self.region_lp(pred_seg, gt_seg, ~merged_output_mask, gt_output_mask, no_weight_mask)
        else:
            region_lp = self.region_lp(pred_seg, gt_seg, ~merged_output_mask, merged_output_mask)

        # create dict containing both results
        region_lp["Total Loss"] = region_lp["Total Loss"] * self.weight

        return region_lp #, pred_output_mask


class MovementConsistencyLoss(nn.Module):
    """
    Class for simultaneous calculation of L1, content/perceptual losses.
    Compares output rgb image at output mask with input rgb image at input mask.

    Semantics: A table that was moved from A to B should still have similar colors at position B as at position A.
    """

    def __init__(self, losses=['1.0_l1', '10.0_content']):
        """
        :param losses:
            loss specification, str of the form: 'lambda_loss'
            lambda is used to weight different losses
            l1 and content/perceptual losses are summed if both are specified
            used in the forward method

        :param ignore_at_scene_editing_masks:
            If true, we do not calculate the loss for the input and output mask of scene editing movements.
            We do this by setting the pred_img equal to the gt_img at these mask positions, s.t. the loss will be 0 at these locations.
            If false (default), we calculate the losses over the whole image as it is and we ignore the masks.
        """

        super().__init__()

        lambdas, loss_names = zip(
            *[loss_name.split("_") for loss_name in losses])  # Parse lambda and loss_names from str
        print("Loss names:", loss_names)
        print("Weight of each loss:", lambdas)
        lambdas = [float(l) for l in lambdas]  # [str] -> [float]

        self.lambdas = lambdas
        self.losses = nn.ModuleList(
            [self.get_loss_from_name(loss_name) for loss_name in loss_names]
        )

    def get_loss_from_name(self, name):
        if name == "l1":
            loss = L1LossWrapper()
        elif name == "content":
            loss = PerceptualLoss()
        else:
            raise ValueError("Invalid loss name in SynthesisLoss: " + name)
        # TODO: If needed, more loss classes can be introduced here later on.

        if torch.cuda.is_available():
            return loss.cuda()
        else:
            return loss

    def forward(self, pred_img, input_img, pred_output_mask, input_mask):

        pred_img = pred_img.clone()
        input_img = input_img.clone()

        # TODO how to vectorize this?

        # img is in format (bs, c, h, w)
        # mask is in format (bs, 1, h, w)
        # In order to use this in one vectorized call we do:
        # mask.squeeze() --> (bs, h, w)
        # img.permute(0,2,3,1) --> (bs, h, w, c)
        # img[mask, :] now accesses the masked pixels at every channel

        pred_img = pred_img.permute(0,2,3,1)
        input_img = input_img.permute(0,2,3,1)

        pred_img = pred_img[pred_output_mask.squeeze(), :]
        input_img = input_img[input_mask.squeeze(), :]

        pred_img = pred_img.permute(0, 3, 1, 2)
        input_img = input_img.permute(0, 3, 1, 2)

        # TODO must reshape in same size...

        '''
        bs = pred_img.shape[0]
        for i in range(bs):
            import matplotlib.pyplot as plt
            plt.imshow(pred_img[i].permute((1, 2, 0)).cpu().detach().numpy())
            plt.show()
            
            plt.imshow(input_img[i].permute((1, 2, 0)).cpu().detach().numpy())
            plt.show()

            print("mask in")
            plt.imshow(input_mask.cpu().detach().numpy())
            plt.show()

            print("mask out")
            plt.imshow(pred_output_mask.cpu().detach().numpy())
            plt.show()
        '''

        # Initialize output dict
        results = {"Total Loss": 0}

        for i, func in enumerate(self.losses):
            # Evaluate each different loss (L1, Content/Perceptual) function with the prediction and target
            out = func(pred_img, input_img)

            # Add the contribution by each loss to the total loss wrt their weights (lambda)
            results["Total Loss"] += out["Total Loss"] * self.lambdas[i]

            # Merge both dicts and store the resulting dict in results
            results = dict(out, **results)

        return results  # Contains individual losses and weighted sum of these


class SynthesisLoss(nn.Module):
    """
    Class for simultaneous calculation of L1, content/perceptual losses.
    Losses to use should be passed as argument.
    """
    def __init__(self, losses=['1.0_l1', '10.0_content'], ignore_at_scene_editing_masks=False):
        """
        :param losses: 
            loss specification, str of the form: 'lambda_loss'
            lambda is used to weight different losses
            l1 and content/perceptual losses are summed if both are specified
            used in the forward method

        :param ignore_at_scene_editing_masks:
            If true, we do not calculate the loss for the input and output mask of scene editing movements.
            We do this by setting the pred_img equal to the gt_img at these mask positions, s.t. the loss will be 0 at these locations.
            If false (default), we calculate the losses over the whole image as it is and we ignore the masks.
        """

        super().__init__()

        lambdas, loss_names = zip(*[loss_name.split("_") for loss_name in losses]) # Parse lambda and loss_names from str
        print("Loss names:", loss_names)
        print("Weight of each loss:", lambdas)
        lambdas = [float(l) for l in lambdas] # [str] -> [float]

        self.lambdas = lambdas
        self.losses = nn.ModuleList(
            [self.get_loss_from_name(loss_name) for loss_name in loss_names]
        )
        self.ignore_at_scene_editing_masks = ignore_at_scene_editing_masks

    def get_loss_from_name(self, name):
        if name == "l1":
            loss = L1LossWrapper()
        elif name == "content":
            loss = PerceptualLoss()
        else:
            raise ValueError("Invalid loss name in SynthesisLoss: " + name)
        # TODO: If needed, more loss classes can be introduced here later on.

        if torch.cuda.is_available():
            return loss.cuda()
        else:
            return loss

    def forward(self, pred_img, gt_img, input_mask=None, gt_output_mask=None):
        """
        For each loss function provided, evaluate the function with prediction and target.
        Results of individual functions along with the total loss returned in a dictionary.
        
        :param pred_img: NVS image outputted from the generator
            used for loss calculation/metric evaluation
        :param gt_img: GT image for the novel view
            used for loss calculation/metric evaluation
        :param input_mask: from which pixels were objects moved to another location
        :param gt_output_mask: to which pixels should objects be moved to after applying movements
        """

        if self.ignore_at_scene_editing_masks:
            if input_mask is None or gt_output_mask is None:
                raise ValueError("You wanted to ignore SynthesisLoss at scene editing masks, but provided no masks in the forward pass.")

            pred_img = pred_img.clone()
            gt_img = gt_img.clone()
            pred_img *= ~input_mask
            pred_img *= ~gt_output_mask
            gt_img *= ~input_mask
            gt_img *= ~gt_output_mask

            '''
            bs = pred_img.shape[0]
            for i in range(bs):
                import matplotlib.pyplot as plt
                plt.imshow(pred_img[i].permute((1, 2, 0)).cpu().detach().numpy())
                plt.show()

                print("mask in")
                plt.imshow(in_mask.cpu().detach().numpy())
                plt.show()

                print("mask out")
                plt.imshow(out_mask.cpu().detach().numpy())
                plt.show()
            '''


        # Initialize output dict
        results = {"Total Loss": 0}

        for i, func in enumerate(self.losses):
            # Evaluate each different loss (L1, Content/Perceptual) function with the prediction and target
            out = func(pred_img, gt_img)

            # Add the contribution by each loss to the total loss wrt their weights (lambda)
            results["Total Loss"] += out["Total Loss"] * self.lambdas[i]
            
            # Merge both dicts and store the resulting dict in results
            results = dict(out, **results) 

        return results # Contains individual losses and weighted sum of these


class QualityMetrics(nn.Module):
    """
    Class for simultaneous calculation of known image quality metrics PSNR, SSIM.
    Metrics to use should be passed as argument.

    PSNR, SSIM will be calculated for both rgb and seg predictions.
    """
    def __init__(self, metrics=["PSNR", "SSIM"], ignore_rgb_at_scene_editing_masks=True):
        super().__init__()

        print("Metric names:", *metrics)

        self.metrics = nn.ModuleList(
            [self.get_metric_from_name(metric) for metric in metrics]
        )

        self.ignore_rgb_at_scene_editing_masks = ignore_rgb_at_scene_editing_masks

    def get_metric_from_name(self, name):
        if name == "PSNR":
            metric = PSNR()
        elif name == "SSIM":
            metric = SSIM()
        else:
            raise ValueError("Invalid metric name in QualityMetrics: " + name)
        # TODO: If needed, more metric classes can be introduced here later on.

        if torch.cuda.is_available():
            return metric.cuda()

    def forward(self, pred_img, gt_img, pred_seg, gt_seg, gt_output_mask=None, input_mask=None):
        """
        For each metric function provided, evaluate the function with prediction and target.
        Output is returned in "results" dict.
        
        :param pred_img: NVS image outputted from the generator
            used for loss calculation/metric evaluation
        :param gt_img: GT image for the novel view
            used for loss calculation/metric evaluation
        """

        # Initialize output dict
        results = {}

        if self.ignore_rgb_at_scene_editing_masks and gt_output_mask is not None and input_mask is not None:
            # set pred_img and gt_img to zero at mask regions
            pred_img = pred_img.clone()
            gt_img = gt_img.clone()
            pred_img *= ~input_mask
            pred_img *= ~gt_output_mask
            gt_img *= ~input_mask
            gt_img *= ~gt_output_mask

        for func in self.metrics:
            if isinstance(func, PSNR) or isinstance(func, SSIM):
                # calculate for rgb and add prefix "rgb" to result
                out_rgb = func(pred_img, gt_img)
                rgb_key = next(iter(out_rgb))
                out_rgb["rgb_"+str(rgb_key)] = out_rgb.pop(rgb_key)

                # calculate for seg and add prefix "seg" to result
                out_seg = func(pred_seg, gt_seg)
                seg_key = next(iter(out_seg))
                out_seg["seg_" + str(seg_key)] = out_seg.pop(seg_key)

                # add to overall output dict
                results.update(out_rgb)
                results.update(out_seg)
            else:
                raise ValueError("Invalid metric in QualityMetrics: " + func)

        return results # Contains individual metric measurements
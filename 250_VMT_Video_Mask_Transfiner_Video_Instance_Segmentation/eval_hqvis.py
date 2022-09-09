import argparse


from pycocotools.ytvos import YTVOS
from pycocotools.ytvoseval import YTVOSeval

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def ytvos_eval(result_file, result_types, ytvos, get_boundary_out, max_dets=(100, 300, 1000)):
    
    ytvos = YTVOS(ytvos, get_boundary=get_boundary_out)
    assert isinstance(ytvos, YTVOS)

    if len(ytvos.anns) == 0:
        print("Annotations does not exist")
        return
    
    assert result_file.endswith('.json')
    ytvos_dets = ytvos.loadRes(result_file)

    vid_ids = ytvos.getVidIds()
    for res_type in result_types:
        iou_type = res_type
        ytvosEval = YTVOSeval(ytvos, ytvos_dets, iou_type)
        ytvosEval.params.vidIds = vid_ids
        if res_type == 'proposal':
            ytvosEval.params.useCats = 0
            ytvosEval.params.maxDets = list(max_dets)
        ytvosEval.evaluate()
        ytvosEval.accumulate()
        ytvosEval.summarize()

def main(args):
    result_file = args.save_path
    ytvos = 'ytvos'
    ytvos_eval(result_file, ['boundary'], 'ytvis/annotations/ytvis_hq-test.json', True, max_dets=(100, 300, 1000))
    ytvos_eval(result_file, ['segm'], 'ytvis/annotations/ytvis_hq-test.json', False, max_dets=(100, 300, 1000))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('inference script')
    parser.add_argument('--save-path')
    args = parser.parse_args()
    main(args)

class_names = [
    'motorcycle',
    'bus',
    'truck',
    'car',
    'airplane',
    
    'bird',
    'sheep',
    'elephant',
    'zebra',
    'horse',
    'cow',
    'bear',
    'giraffe',
]

class_indices = {
    # Vehicles
    'motorcycle': 0, # ImageNet
    'bus': 1, # ImageNet
    'truck': 2, # ImageNet
    'car': 3, # ImageNet/P3D
    'airplane': 4, # ImageNet/P3D
    
    # Animals
    'bird': 5, # CUB
    'sheep': 6, # ImageNet
    'elephant': 7, # ImageNet
    'zebra': 8, # ImageNet
    'horse': 9, # ImageNet
    'cow': 10, # ImageNet
    'bear': 11, # ImageNet
    'giraffe': 12, # ImageNet
}

default_cache_directories = {
    'motorcycle': 'imagenet_motorcycle',
    'bus': 'imagenet_bus',
    'truck': 'imagenet_truck',
    'car': 'imagenet_car',
    'airplane': 'imagenet_airplane',
    
    'bird': 'cub',
    'sheep': 'imagenet_sheep',
    'elephant': 'imagenet_elephant',
    'zebra': 'imagenet_zebra',
    'horse': 'imagenet_horse',
    'cow': 'imagenet_cow',
    'bear': 'imagenet_bear',
    'giraffe': 'imagenet_giraffe',
}

dataset_to_class_name = {v: k for k, v in default_cache_directories.items()}
dataset_to_class_name['p3d_car'] = 'car'
dataset_to_class_name['p3d_airplane'] = 'airplane'
# *** Add custom datasets here ***
# dataset_to_class_name['your_custom_dataset_name'] = 'category_name'

imagenet_synsets = {
    'motorcycle': ['n03790512', 'n03791053', 'n04466871'],
    'bus': ['n04146614', 'n02924116'],
    'truck': ['n03345487', 'n03417042', 'n03796401'],
    'car': ['n02814533', 'n02958343', 'n03498781', 'n03770085', 'n03770679', 'n03930630', 'n04037443', 'n04166281', 'n04285965'],
    'airplane': ['n02690373', 'n02691156', 'n03335030', 'n04012084'],
    
    'sheep': ['n10588074', 'n02411705', 'n02413050', 'n02412210'],
    'elephant': ['n02504013', 'n02504458'],
    'zebra': ['n02391049', 'n02391234', 'n02391373', 'n02391508'],
    'horse': ['n02381460', 'n02374451'],
    'cow': ['n01887787', 'n02402425'],
    'bear': ['n02132136', 'n02133161', 'n02131653', 'n02134084'],
    'giraffe': ['n02439033'],
}

vg3k_class_set = ['wing', 'engine', 'fender', 'seat', 'gas_tank', 'handlebar', 
                  'door', 'bumper', 'grillroom', 'license_plate', 'wheel', 'window', 'windshield', 'mirror', 'light', 'headlight',
                  'landing_gear', 'feather', 'tail', 'leg', 'foot', 'hoof',
                  'neck', 'mane', 'head', 'face', 'mouth', 'nose', 'cockpit', 'trunk', 'horn', 'ear', 'eye', 'beak']
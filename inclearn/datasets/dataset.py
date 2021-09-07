import os.path as osp
import numpy as np
import glob

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torchvision import datasets, transforms
import torch


def get_datasets(dataset_names):
    return [get_dataset(dataset_name) for dataset_name in dataset_names.split("-")]


def get_dataset(dataset_name):
    if dataset_name == "cifar10":
        return iCIFAR10
    elif dataset_name == "cifar100":
        return iCIFAR100
    elif "imagenet100" in dataset_name:
        return iImageNet100
    elif dataset_name == "imagenet":
        return iImageNet
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


class DataHandler:
    base_dataset = None
    train_transforms = []
    common_transforms = [ToTensorV2()]
    class_order = None


class iCIFAR10(DataHandler):
    base_dataset_cls = datasets.cifar.CIFAR10
    transform_type = 'torchvision'
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    def __init__(self, data_folder, train, is_fine_label=False):
        self.base_dataset = self.base_dataset_cls(data_folder, train=train, download=True)
        self.data = self.base_dataset.data
        self.targets = self.base_dataset.targets
        self.n_cls = 10

    @property
    def is_proc_inc_data(self):
        return False

    @classmethod
    def class_order(cls, trial_i):
        return [4, 0, 2, 5, 8, 3, 1, 6, 9, 7]


class iCIFAR100(iCIFAR10):
    label_list = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy',
        'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish',
        'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange',
        'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk',
        'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
        'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale',
        'willow_tree', 'wolf', 'woman', 'worm'
    ]
    base_dataset_cls = datasets.cifar.CIFAR100
    transform_type = 'torchvision'
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    def __init__(self, data_folder, train, is_fine_label=False):
        self.base_dataset = self.base_dataset_cls(data_folder, train=train, download=True)
        self.data = self.base_dataset.data
        self.targets = self.base_dataset.targets
        self.n_cls = 100
        self.transform_type = 'torchvision'

    @property
    def is_proc_inc_data(self):
        return False

    @classmethod
    def class_order(cls, trial_i):
        if trial_i == 0:
            return [
                62, 54, 84, 20, 94, 22, 40, 29, 78, 27, 26, 79, 17, 76, 68, 88, 3, 19, 31, 21, 33, 60, 24, 14, 6, 10,
                16, 82, 70, 92, 25, 5, 28, 9, 61, 36, 50, 90, 8, 48, 47, 56, 11, 98, 35, 93, 44, 64, 75, 66, 15, 38, 97,
                42, 43, 12, 37, 55, 72, 95, 18, 7, 23, 71, 49, 53, 57, 86, 39, 87, 34, 63, 81, 89, 69, 46, 2, 1, 73, 32,
                67, 91, 0, 51, 83, 13, 58, 80, 74, 65, 4, 30, 45, 77, 99, 85, 41, 96, 59, 52
            ]
        elif trial_i == 1:
            return [
                68, 56, 78, 8, 23, 84, 90, 65, 74, 76, 40, 89, 3, 92, 55, 9, 26, 80, 43, 38, 58, 70, 77, 1, 85, 19, 17,
                50, 28, 53, 13, 81, 45, 82, 6, 59, 83, 16, 15, 44, 91, 41, 72, 60, 79, 52, 20, 10, 31, 54, 37, 95, 14,
                71, 96, 98, 97, 2, 64, 66, 42, 22, 35, 86, 24, 34, 87, 21, 99, 0, 88, 27, 18, 94, 11, 12, 47, 25, 30,
                46, 62, 69, 36, 61, 7, 63, 75, 5, 32, 4, 51, 48, 73, 93, 39, 67, 29, 49, 57, 33
            ]
        elif trial_i == 2:  #PODNet
            return [
                87, 0, 52, 58, 44, 91, 68, 97, 51, 15, 94, 92, 10, 72, 49, 78, 61, 14, 8, 86, 84, 96, 18, 24, 32, 45,
                88, 11, 4, 67, 69, 66, 77, 47, 79, 93, 29, 50, 57, 83, 17, 81, 41, 12, 37, 59, 25, 20, 80, 73, 1, 28, 6,
                46, 62, 82, 53, 9, 31, 75, 38, 63, 33, 74, 27, 22, 36, 3, 16, 21, 60, 19, 70, 90, 89, 43, 5, 42, 65, 76,
                40, 30, 23, 85, 2, 95, 56, 48, 71, 64, 98, 13, 99, 7, 34, 55, 54, 26, 35, 39
            ]
        elif trial_i == 3:  #PODNet
            return [
                58, 30, 93, 69, 21, 77, 3, 78, 12, 71, 65, 40, 16, 49, 89, 46, 24, 66, 19, 41, 5, 29, 15, 73, 11, 70,
                90, 63, 67, 25, 59, 72, 80, 94, 54, 33, 18, 96, 2, 10, 43, 9, 57, 81, 76, 50, 32, 6, 37, 7, 68, 91, 88,
                95, 85, 4, 60, 36, 22, 27, 39, 42, 34, 51, 55, 28, 53, 48, 38, 17, 83, 86, 56, 35, 45, 79, 99, 84, 97,
                82, 98, 26, 47, 44, 62, 13, 31, 0, 75, 14, 52, 74, 8, 20, 1, 92, 87, 23, 64, 61
            ]
        elif trial_i == 4:  #PODNet
            return [
                71, 54, 45, 32, 4, 8, 48, 66, 1, 91, 28, 82, 29, 22, 80, 27, 86, 23, 37, 47, 55, 9, 14, 68, 25, 96, 36,
                90, 58, 21, 57, 81, 12, 26, 16, 89, 79, 49, 31, 38, 46, 20, 92, 88, 40, 39, 98, 94, 19, 95, 72, 24, 64,
                18, 60, 50, 63, 61, 83, 76, 69, 35, 0, 52, 7, 65, 42, 73, 74, 30, 41, 3, 6, 53, 13, 56, 70, 77, 34, 97,
                75, 2, 17, 93, 33, 84, 99, 51, 62, 87, 5, 15, 10, 78, 67, 44, 59, 85, 43, 11
            ]


class DataHandler:
    base_dataset = None
    train_transforms = []
    common_transforms = [ToTensorV2()]
    class_order = None


class iImageNet(DataHandler):
    base_dataset_cls = datasets.ImageFolder
    transform_type = 'albumentations'
    if transform_type == 'albumentations':
        train_transforms = A.Compose([
            A.RandomResizedCrop(224, 224),
            A.HorizontalFlip(),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2()
        ])
        test_transforms = A.Compose([
            A.Resize(256, 256),
            A.CenterCrop(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2()
        ])
    else:
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(brightness=63 / 255),
        ])
        test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __init__(self, data_folder, train, is_fine_label=False):
        if train is True:
            self.base_dataset = self.base_dataset_cls(osp.join(data_folder, "train"))
        else:
            self.base_dataset = self.base_dataset_cls(osp.join(data_folder, "val"))
        self.data, self.targets = zip(*self.base_dataset.samples)
        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
        self.n_cls = 1000

    @property
    def is_proc_inc_data(self):
        return False

    @classmethod
    def class_order(cls, trial_i):
        return [
            54, 7, 894, 512, 126, 337, 988, 11, 284, 493, 133, 783, 192, 979, 622, 215, 240, 548, 238, 419, 274, 108,
            928, 856, 494, 836, 473, 650, 85, 262, 508, 590, 390, 174, 637, 288, 658, 219, 912, 142, 852, 160, 704, 289,
            123, 323, 600, 542, 999, 634, 391, 761, 490, 842, 127, 850, 665, 990, 597, 722, 748, 14, 77, 437, 394, 859,
            279, 539, 75, 466, 886, 312, 303, 62, 966, 413, 959, 782, 509, 400, 471, 632, 275, 730, 105, 523, 224, 186,
            478, 507, 470, 906, 699, 989, 324, 812, 260, 911, 446, 44, 765, 759, 67, 36, 5, 30, 184, 797, 159, 741, 954,
            465, 533, 585, 150, 101, 897, 363, 818, 620, 824, 154, 956, 176, 588, 986, 172, 223, 461, 94, 141, 621, 659,
            360, 136, 578, 163, 427, 70, 226, 925, 596, 336, 412, 731, 755, 381, 810, 69, 898, 310, 120, 752, 93, 39,
            326, 537, 905, 448, 347, 51, 615, 601, 229, 947, 348, 220, 949, 972, 73, 913, 522, 193, 753, 921, 257, 957,
            691, 155, 820, 584, 948, 92, 582, 89, 379, 392, 64, 904, 169, 216, 694, 103, 410, 374, 515, 484, 624, 409,
            156, 455, 846, 344, 371, 468, 844, 276, 740, 562, 503, 831, 516, 663, 630, 763, 456, 179, 996, 936, 248,
            333, 941, 63, 738, 802, 372, 828, 74, 540, 299, 750, 335, 177, 822, 643, 593, 800, 459, 580, 933, 306, 378,
            76, 227, 426, 403, 322, 321, 808, 393, 27, 200, 764, 651, 244, 479, 3, 415, 23, 964, 671, 195, 569, 917,
            611, 644, 707, 355, 855, 8, 534, 657, 571, 811, 681, 543, 313, 129, 978, 592, 573, 128, 243, 520, 887, 892,
            696, 26, 551, 168, 71, 398, 778, 529, 526, 792, 868, 266, 443, 24, 57, 15, 871, 678, 745, 845, 208, 188,
            674, 175, 406, 421, 833, 106, 994, 815, 581, 676, 49, 619, 217, 631, 934, 932, 568, 353, 863, 827, 425, 420,
            99, 823, 113, 974, 438, 874, 343, 118, 340, 472, 552, 937, 0, 10, 675, 316, 879, 561, 387, 726, 255, 407,
            56, 927, 655, 809, 839, 640, 297, 34, 497, 210, 606, 971, 589, 138, 263, 587, 993, 973, 382, 572, 735, 535,
            139, 524, 314, 463, 895, 376, 939, 157, 858, 457, 935, 183, 114, 903, 767, 666, 22, 525, 902, 233, 250, 825,
            79, 843, 221, 214, 205, 166, 431, 860, 292, 976, 739, 899, 475, 242, 961, 531, 110, 769, 55, 701, 532, 586,
            729, 253, 486, 787, 774, 165, 627, 32, 291, 962, 922, 222, 705, 454, 356, 445, 746, 776, 404, 950, 241, 452,
            245, 487, 706, 2, 137, 6, 98, 647, 50, 91, 202, 556, 38, 68, 649, 258, 345, 361, 464, 514, 958, 504, 826,
            668, 880, 28, 920, 918, 339, 315, 320, 768, 201, 733, 575, 781, 864, 617, 171, 795, 132, 145, 368, 147, 327,
            713, 688, 848, 690, 975, 354, 853, 148, 648, 300, 436, 780, 693, 682, 246, 449, 492, 162, 97, 59, 357, 198,
            519, 90, 236, 375, 359, 230, 476, 784, 117, 940, 396, 849, 102, 122, 282, 181, 130, 467, 88, 271, 793, 151,
            847, 914, 42, 834, 521, 121, 29, 806, 607, 510, 837, 301, 669, 78, 256, 474, 840, 52, 505, 547, 641, 987,
            801, 629, 491, 605, 112, 429, 401, 742, 528, 87, 442, 910, 638, 785, 264, 711, 369, 428, 805, 744, 380, 725,
            480, 318, 997, 153, 384, 252, 985, 538, 654, 388, 100, 432, 832, 565, 908, 367, 591, 294, 272, 231, 213,
            196, 743, 817, 433, 328, 970, 969, 4, 613, 182, 685, 724, 915, 311, 931, 865, 86, 119, 203, 268, 718, 317,
            926, 269, 161, 209, 807, 645, 513, 261, 518, 305, 758, 872, 58, 65, 146, 395, 481, 747, 41, 283, 204, 564,
            185, 777, 33, 500, 609, 286, 567, 80, 228, 683, 757, 942, 134, 673, 616, 960, 450, 350, 544, 830, 736, 170,
            679, 838, 819, 485, 430, 190, 566, 511, 482, 232, 527, 411, 560, 281, 342, 614, 662, 47, 771, 861, 692, 686,
            277, 373, 16, 946, 265, 35, 9, 884, 909, 610, 358, 18, 737, 977, 677, 803, 595, 135, 458, 12, 46, 418, 599,
            187, 107, 992, 770, 298, 104, 351, 893, 698, 929, 502, 273, 20, 96, 791, 636, 708, 267, 867, 772, 604, 618,
            346, 330, 554, 816, 664, 716, 189, 31, 721, 712, 397, 43, 943, 804, 296, 109, 576, 869, 955, 17, 506, 963,
            786, 720, 628, 779, 982, 633, 891, 734, 980, 386, 365, 794, 325, 841, 878, 370, 695, 293, 951, 66, 594, 717,
            116, 488, 796, 983, 646, 499, 53, 1, 603, 45, 424, 875, 254, 237, 199, 414, 307, 362, 557, 866, 341, 19,
            965, 143, 555, 687, 235, 790, 125, 173, 364, 882, 727, 728, 563, 495, 21, 558, 709, 719, 877, 352, 83, 998,
            991, 469, 967, 760, 498, 814, 612, 715, 290, 72, 131, 259, 441, 924, 773, 48, 625, 501, 440, 82, 684, 862,
            574, 309, 408, 680, 623, 439, 180, 652, 968, 889, 334, 61, 766, 399, 598, 798, 653, 930, 149, 249, 890, 308,
            881, 40, 835, 577, 422, 703, 813, 857, 995, 602, 583, 167, 670, 212, 751, 496, 608, 84, 639, 579, 178, 489,
            37, 197, 789, 530, 111, 876, 570, 700, 444, 287, 366, 883, 385, 536, 460, 851, 81, 144, 60, 251, 13, 953,
            270, 944, 319, 885, 710, 952, 517, 278, 656, 919, 377, 550, 207, 660, 984, 447, 553, 338, 234, 383, 749,
            916, 626, 462, 788, 434, 714, 799, 821, 477, 549, 661, 206, 667, 541, 642, 689, 194, 152, 981, 938, 854,
            483, 332, 280, 546, 389, 405, 545, 239, 896, 672, 923, 402, 423, 907, 888, 140, 870, 559, 756, 25, 211, 158,
            723, 635, 302, 702, 453, 218, 164, 829, 247, 775, 191, 732, 115, 331, 901, 416, 873, 754, 900, 435, 762,
            124, 304, 329, 349, 295, 95, 451, 285, 225, 945, 697, 417
        ]


class iImageNet100(DataHandler):

    base_dataset_cls = datasets.ImageFolder
    transform_type = 'albumentations'
    if transform_type == 'albumentations':
        train_transforms = A.Compose([
            A.RandomResizedCrop(224, 224),
            A.HorizontalFlip(),
            # A.ColorJitter(brightness=63 / 255),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        test_transforms = A.Compose([
            A.Resize(256, 256),
            A.CenterCrop(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(brightness=63 / 255),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __init__(self, data_folder, train, is_fine_label=False):
        if train is True:
            self.base_dataset = self.base_dataset_cls(osp.join(data_folder, "train"))
        else:
            self.base_dataset = self.base_dataset_cls(osp.join(data_folder, "val"))

        self.data, self.targets = zip(*self.base_dataset.samples)
        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
        self.n_cls = 100

    @property
    def is_proc_inc_data(self):
        return False

    @classmethod
    def class_order(cls, trial_i):
        return [
            68, 56, 78, 8, 23, 84, 90, 65, 74, 76, 40, 89, 3, 92, 55, 9, 26, 80, 43, 38, 58, 70, 77, 1, 85, 19, 17, 50,
            28, 53, 13, 81, 45, 82, 6, 59, 83, 16, 15, 44, 91, 41, 72, 60, 79, 52, 20, 10, 31, 54, 37, 95, 14, 71, 96,
            98, 97, 2, 64, 66, 42, 22, 35, 86, 24, 34, 87, 21, 99, 0, 88, 27, 18, 94, 11, 12, 47, 25, 30, 46, 62, 69,
            36, 61, 7, 63, 75, 5, 32, 4, 51, 48, 73, 93, 39, 67, 29, 49, 57, 33
        ]

from continuum.datasets import CIFAR100 as ICIFAR100
from continuum import ClassIncremental
import torch
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import timm
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.cluster import KMeans
import argparse

parser = argparse.ArgumentParser(description="Process some integers.")

parser.add_argument("-s", "--scale_list", type=float, nargs='+', default=[1.0], help="List of scales to be used.")
parser.add_argument("-n", "--n_clusters", type=int, default=1, help="Number of clusters.")
args = parser.parse_args()

SCALE_LIST = args.scale_list
n_clusters = args.n_clusters
print(f"Scale list: {SCALE_LIST}")
print(f"Number of clusters: {n_clusters}")

train_ds = ICIFAR100(data_path="./data", train=True, download=True)
test_ds = ICIFAR100(data_path="./data", train=False, download=True)

scenario_train = ClassIncremental(train_ds, increment=10, initial_increment=10,
                                  transformations=[ToTensor()])
scenario_test = ClassIncremental(test_ds, increment=10, initial_increment=10, transformations=[ToTensor()])


def multiProcess(original_img_batch):
    feat_list = []
    for s in SCALE_LIST:
        img_batch = original_img_batch.clone()
        original_height, original_width = img_batch.shape[2], img_batch.shape[3]
        new_height, new_width = int(original_height * s), int(original_width * s)

        if s > 1:
            # Create a larger image filled with white color
            larger_img = torch.ones((img_batch.shape[0], img_batch.shape[1], new_height, new_width))
            # Calculate the position to place the original image
            top = (new_height - original_height) // 2
            left = (new_width - original_width) // 2
            larger_img[:, :, top:top + original_height, left:left + original_width] = img_batch
            img_batch = larger_img.cuda()
        else:
            # Calculate the new region's top-left and bottom-right coordinates
            top = (original_height - new_height) // 2
            left = (original_width - new_width) // 2
            bottom = top + new_height
            right = left + new_width

            # Crop the new region
            img_batch = img_batch[:, :, top:bottom, left:right]

        img_batch = F.interpolate(img_batch, size=(224, 224), mode='bilinear', align_corners=False)
        out = F.normalize(vit_b_16.forward_features(img_batch)[:, 0].detach())
        feat_list.append(out)
        img_feats_agg = [img_feat.unsqueeze(0) for img_feat in feat_list]
        img_feats_agg = torch.cat(img_feats_agg, dim=0)
        img_feats_agg = torch.mean(img_feats_agg, dim=0)
        img_feats_agg = F.normalize(img_feats_agg, p=2, dim=1
                                    )
        img_feats_agg = img_feats_agg.cpu().numpy()
    return img_feats_agg


if __name__ == '__main__':
    print(torch.cuda.is_available())
    vit_b_16 = timm.create_model("vit_base_patch16_224_in21k", pretrained=True).cuda()
    class_mean_set = []
    accuracy_history = []
    for task_id, train_dataset in enumerate(scenario_train):
        train_loader = DataLoader(train_dataset, batch_size=128)
        X = []
        y = []
        for (img_batch, label, t) in tqdm(train_loader):
            img_batch = img_batch.cuda()
            with torch.no_grad():
                out = multiProcess(img_batch)
            X.append(out)
            y.append(label)
        X = np.concatenate(X)
        y = np.concatenate(y)
        for i in range(task_id * 10, (task_id + 1) * 10):
            image_class_mask = (y == i)
            kmeans = KMeans(n_clusters=n_clusters, random_state=0, init="k-means++", n_init="auto").fit(
                X[image_class_mask])
            for center in kmeans.cluster_centers_:
                class_mean_set.append(center)

        test_ds = scenario_test[:task_id + 1]
        test_loader = DataLoader(test_ds, batch_size=128)
        correct, total = 0, 0
        for (img_batch, label, t) in tqdm(test_loader, desc="训练进度"):
            img_batch = img_batch.cuda()
            with torch.no_grad():
                out = multiProcess(img_batch)

            predictions = []
            for single_image in out:
                distance = single_image - class_mean_set
                norm = np.linalg.norm(distance, ord=2, axis=1)
                pred = np.argmin(norm)
                pred = (pred) // n_clusters
                predictions.append(pred)
            predictions = torch.tensor(predictions)
            correct += (predictions.cpu() == label.cpu()).sum()
            total += label.shape[0]
        print(f"Accuracy at {task_id} {correct / total}")
        accuracy_history.append(correct / total)

    print(accuracy_history)
    print(f"average incremental accuracy {round(np.mean(np.array(accuracy_history)) * 100, 2)} ")

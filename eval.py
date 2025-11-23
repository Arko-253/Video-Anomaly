import torch
import numpy as np
from torch.utils.data import DataLoader
from models.aa_rae import AA_RAE
from utils.dataset import UCSDFrameDataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

# load test dataset
test_dataset = UCSDFrameDataset("data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# load model
model = AA_RAE().to(device)
model.load_state_dict(torch.load("checkpoints/ucsd_epoch_30.pth"))
model.eval()

scores = []

criterion = torch.nn.MSELoss(reduction='mean')

with torch.no_grad():
    for frame in tqdm(test_loader):
        frame = frame.to(device)
        recon = model(frame)

        loss = criterion(recon, frame).item()
        scores.append(loss)

scores = np.array(scores)

# load ground truth
labels = np.load("data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/test_labels.npy")


auc = roc_auc_score(labels, scores)
print("AUC on UCSD Ped2:", auc)

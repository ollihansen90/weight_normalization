import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms


dataset_test = dset.EMNIST(
    root="datasets",
    split="byclass",
    train=False,
    transform=transforms.Compose([transforms.ToTensor()]),
    download=True
)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=2**11, shuffle=True, num_workers=4)
model = torch.load("models/model_1631096381.pt")

sample, _ = next(iter(dataloader_test))
for layer in model.parameters():
    print(layer)

# --- Ermittle mittlere Accuracy und Loss für MLP_BN mit 100 Trainingsepochen ---
"""filepath = "MLP_BN/MLP_BN_100.pt"

datadict = torch.load(filepath) # ['accliste_test', 'accliste_train', 'lossliste', 'mininorms', 'maxinorms']

datadict_mean = {}
for key in ['accliste_test', 'accliste_train', 'lossliste']:
    datadict_mean[key] = datadict[key].mean(dim=0)
    print(datadict_mean[key].shape)
torch.save(datadict_mean, "MLP_BN/MLP_BN_100_mean.pt")"""

# --- Plotte mittlere Outnorm (und Streuung) ---
"""modelstamp = [  "1629963509", 
                "1629963934", 
                "1629964353", 
                "1629964776", 
                "1629965199", 
                "1629965624", 
                "1629966040", 
                "1629966461", 
                "1629966891", 
                "1629967324", 
                "1629967750", 
                "1629968186",
                "1629968612",
                "1629969032",
                "1629969451"
            ]
torch.set_printoptions(edgeitems=10_000)
for idx, stamp in enumerate(modelstamp):
    print(stamp)
    model = torch.load("models/model_"+stamp+".pt")
    for ii, param in enumerate(model.named_parameters()):
        
        #idx = int(param[0].split(".")[2])
        if "outnorm" in param[0]:
            #print(idx, param)
            print(idx, "\t", torch.mean(param[1]).item(), "\t", torch.std(param[1]).item())"""

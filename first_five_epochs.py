import torch
import torch.nn.functional as F
from MLP import MLP, MLP_normed, Trainer, NetworkGenerator, Normalizer
#from dataloader import Dataloader
#from dataset import MNIST_data

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from lamb import Lamb

import matplotlib.pyplot as plt
from datetime import timedelta
from datetime import datetime as dt

device = "cuda" if torch.cuda.is_available() else "cpu"
plotstuff = 1
savemodel = 1
save_every = 5
start_epoch = 0

#labellist = [3,4,7]
labellist = list(range(62))
num_classes = 62
#dataset = MNIST_data(labels=labellist)
#dataloader = Dataloader(dataset, labels=labellist)
dataset_train = dset.EMNIST(
    root="datasets",
    split="byclass",
    train=True,
    transform=transforms.Compose([transforms.ToTensor()]),
    download=True
)
dataset_test = dset.EMNIST(
    root="datasets",
    split="byclass",
    train=False,
    transform=transforms.Compose([transforms.ToTensor()]),
    download=True
)
n_data_train = len(dataset_train)
n_data_test = len(dataset_test)
batch_size = 2**11
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=4)
print(n_data_train)
print(n_data_test)

normalizer = Normalizer(device=device, load=True)#.to(device)
"""for img, label in dataloader_train:
    normalizer.estimate(img.flatten(-2,-1).to(device))
normalizer.estimate_done()
normalizer.save_imgs()"""

lr = 1e-4 # LAMB: 5e-5
betas = (0.9, 0.999)
#optimizer = torch.optim.SGD(model.parameters(), lr=lr)#, betas=betas)

n_epochs = 5

#params = [5e-3, 2e-3, 1e-3, 5e-4]#[::-1] # next evtl. n_epoch=200 
n_params = 1 #len(params)
params = NetworkGenerator(n=n_params) #int(n_params/2))

lossliste = torch.zeros(n_params, n_epochs).to(device)
accliste_train = torch.zeros(n_params, n_epochs).to(device)
accliste_test = torch.zeros(n_params, n_epochs).to(device)
loss_func = torch.nn.CrossEntropyLoss()
for param_idx, param in enumerate(params):
    #lr = param
    starttime = dt.now().timestamp()
    model = Trainer(param, normalizer).to(device)
    #model = torch.load("models/"+modelnames[param_idx]+".pt")
    print(sum([params.numel() for params in model.parameters()]))
    print("Net", param_idx)
    
    with open("where.txt", "a+") as file:
        file.write("--- Net "+str(param)+", "+ str(round(starttime)) + 70*"-"+"\n")
    maxinorms = torch.zeros(n_epochs,4)
    mininorms = torch.zeros(n_epochs,4)
    counter = 0
    for epoch in range(start_epoch, n_epochs+start_epoch):
        if epoch==0: # warmup
            #optimizer = Lamb(model.parameters(), lr=lr, betas=betas)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            #optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        if epoch==1:
            for g in optimizer.param_groups:
                g["lr"]= lr
        epochstart = dt.now().timestamp()
        total_loss = 0
        acc_train = 0
        acc_test = 0

        # Training
        model.train()
        returnednorms = None
        for img, labels in dataloader_train:
            #img, labels = batch
            img, labels = img.to(device), labels.to(device)
            #print(labels[0])
            #labelsmat = F.one_hot(labels, num_classes=10).to(device)
            output = model(img)
            #print(output.shape)
            #loss = torch.sum((output-labelsmat)**2)
            #loss = F.cross_entropy(output, labels)
            loss = loss_func(output, labels)
            acc_train += torch.sum(torch.argmax(output, dim=-1)==labels)#.item()

            optimizer.zero_grad()
            loss.backward()
            #optimizer.step()
            if epoch==1:
                for name, param in model.named_parameters():
                    #if "1" in name or "4" in name or "7" in name:
                    print(name, "\t", torch.mean(param).item(), "\t", torch.std(param).item())
                model.normalize()
                print("\n")
                for name, param in model.named_parameters():
                    #if "1" in name or "4" in name or "7" in name:
                    print(name, "\t", torch.mean(param).item(), "\t", torch.std(param).item())
                exit()
            
            """
            maxinorms[epoch,:] += returnednorms[:,0].T
            mininorms[epoch,:] += returnednorms[:,1].T"""

            total_loss += loss.detach()
        
        # Testing
        model.eval()
        for img, labels in dataloader_test:
            #img, labels = batch
            img, labels = img.to(device), labels.to(device)
            #print(labels[0])
            #labelsmat = F.one_hot(labels, num_classes=10).to(device)
            output = model(img)
            acc_test += torch.sum(torch.argmax(output, dim=-1)==labels)
        
        acc_train = acc_train.item()/n_data_train
        accliste_train[param_idx, epoch-start_epoch] = acc_train
        acc_test = acc_test.item()/n_data_test
        accliste_test[param_idx, epoch-start_epoch] = acc_test
        lossliste[param_idx, epoch-start_epoch] = total_loss.item()
        plottime = dt.now().timestamp()
        line = "{}. {}\t| acc: {}, {}\t|  loss: {}\t| time: {} \t| time total: {}\t{}\t{}".format(
                str(param_idx+1).rjust(3),
                        str(epoch).rjust(3), 
                                str(round(acc_train, 4)).rjust(6),
                                        str(round(acc_test, 4)).rjust(6), 
                                                    str(round(total_loss.item(),3)).rjust(7), 
                                                                str(round(plottime-epochstart, 2)).rjust(5), 
                                                                                    str(round(plottime-starttime,2)).rjust(8), 
                                                                                        str(dt.fromtimestamp(plottime-starttime).strftime("%H:%M:%S")).rjust(8),
                                                                                        str(dt.now()+timedelta(hours=2))
                                                                                            )

        print(line)
        with open("where.txt", "a+") as file:
            file.write(line+"\n")
        
        if epoch%save_every==0 and savemodel:
            torch.save(model, "models/model_{}.pt".format(epoch))

    if savemodel:
        torch.save(model, "models/model_{}.pt".format(round(starttime)))
        torch.save(model, "models/model.pt")
        print("Model saved, {}".format(round(starttime)))

if plotstuff:
    plt.figure()
    for i in range(n_params):
        plt.plot(lossliste[i,:].cpu())
    #plt.legend(params)
    plt.grid()
    plt.savefig("plots/plot_{}.png".format(round(starttime)))

    plt.figure()
    for i in range(n_params):
        plt.plot(accliste_train[i,:].cpu())
        plt.plot(accliste_test[i,:].cpu())
    paramlist = list(["train_"+str(param), "test_"+str(param)] for param in range(n_params)) #params)
    plt.legend([x for y in paramlist for x in y])
    plt.grid()
    plt.savefig("plots/plot_{}_A.png".format(round(starttime)))
 
    """maxinorms = maxinorms*batch_size/n_data_train
    mininorms = mininorms*batch_size/n_data_train

    plt.figure()
    for i in range(4):
        plt.plot(mininorms[:,i])
    plt.title("Mininorms")
    plt.legend(["Layer 0", "Layer 1", "Layer 2", "Layer 3"])
    plt.savefig("plots/plot_{}_mininorms".format(round(starttime)))

    plt.figure()
    for i in range(4):
        plt.plot(maxinorms[:,i])
    plt.title("Maxinorms")
    plt.legend(["Layer 0", "Layer 1", "Layer 2", "Layer 3"])
    plt.savefig("plots/plot_{}_maxinorms".format(round(starttime)))"""
    

torch.save(accliste_test, "auswertungen/accliste_test_{}.pt".format(round(starttime)))
torch.save(accliste_train, "auswertungen/accliste_train_{}.pt".format(round(starttime)))
torch.save(lossliste, "auswertungen/lossliste_{}.pt".format(round(starttime)))
torch.save({"accliste_test": accliste_test, 
            "accliste_train": accliste_train, 
            "lossliste": lossliste, 
            "mininorms": mininorms,
            "maxinorms": maxinorms}, 
            "auswertungen/auswertung.pt"
            )

out1 = model(img[0].unsqueeze(0))
print(model.network.outnorms)
model.normalize()
out2 = model(img[0].unsqueeze(0))
print(model.network.outnorms)
print(out1.shape)
print(torch.linalg.norm(out1-out2))
print(out1@out2/(torch.linalg.norm(out1)*torch.linalg.norm(out2)))
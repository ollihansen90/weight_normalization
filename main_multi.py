import torch
import torch.nn.functional as F
from MLP import MLP, MLP_normed, Trainer
#from dataloader import Dataloader
#from dataset import MNIST_data

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from lamb import Lamb

import matplotlib.pyplot as plt
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

lr = 5e-5
betas = (0.9, 0.999)
#optimizer = torch.optim.SGD(model.parameters(), lr=lr)#, betas=betas)

n_epochs = 30

params = [1,2,10][::-1] # next evtl. n_epoch=200 
#modelnames = ["model_1619164701", "model_1619172429", "model_1619180155", "model_1619187884", "model_1619195610"]
lossliste = torch.zeros(len(params), n_epochs).to(device)
accliste_train = torch.zeros(len(params), n_epochs).to(device)
accliste_test = torch.zeros(len(params), n_epochs).to(device)
loss_func = torch.nn.CrossEntropyLoss()
for param_idx, param in enumerate(params):
    starttime = dt.now().timestamp()
    model = Trainer().to(device)
    #model = VisualTransformer(inner_dim=p, transformer_depth=1, dim_head=49, attn_heads=3, mlp_dim=49, num_classes=num_classes).to(device)
    #model = torch.load("models/"+modelnames[param_idx]+".pt")
    print(sum([params.numel() for params in model.parameters()]))
    print("lr", param)
    
    with open("where.txt", "a+") as file:
        file.write("--- lr "+str(param)+", "+ str(round(starttime)) + 70*"-"+"\n")
    maxinorms = torch.zeros(n_epochs,4)
    mininorms = torch.zeros(n_epochs,4)
    counter = 0
    for epoch in range(start_epoch, n_epochs+start_epoch):
        if epoch==0: # warmup
            optimizer = Lamb(model.parameters(), lr=5e-5, betas=betas)
            #optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
        if epoch==1:
            for g in optimizer.param_groups:
                g["lr"]= 5e-5
        epochstart = dt.now().timestamp()
        total_loss = 0
        acc_train = 0
        acc_test = 0

        # Training
        model.train()
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
            optimizer.step()
            
            returnednorms = model.normalize(p=param)

            maxinorms[epoch,:] += returnednorms[:,0].T
            mininorms[epoch,:] += returnednorms[:,1].T

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
        line = "{}. {}\t| acc: {}, {}\t|  loss: {}\t| time: {} \t| time total: {}\t{}".format(
                str(param_idx+1).rjust(3),
                        str(epoch).rjust(3), 
                                str(round(acc_train, 4)).rjust(6),
                                        str(round(acc_test, 4)).rjust(6), 
                                                    str(round(total_loss.item(),3)).rjust(7), 
                                                                str(round(plottime-epochstart, 2)).rjust(5), 
                                                                                    str(round(plottime-starttime,2)).rjust(8), 
                                                                                        str(dt.fromtimestamp(plottime-starttime).strftime("%H:%M:%S")).rjust(8)
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
    for i in range(len(params)):
        plt.plot(lossliste[i,:].cpu())
    plt.legend(params)
    plt.savefig("plots/plot_{}.png".format(round(starttime)))

    plt.figure()
    for i in range(len(params)):
        plt.plot(accliste_train[i,:].cpu())
        plt.plot(accliste_test[i,:].cpu())
    paramlist = list(["train_"+str(param), "test_"+str(param)] for param in params)
    plt.legend([x for y in paramlist for x in y])
    plt.savefig("plots/plot_{}_A.png".format(round(starttime)))
 
    maxinorms = maxinorms*batch_size/n_data_train
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
    plt.savefig("plots/plot_{}_maxinorms".format(round(starttime)))
    

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

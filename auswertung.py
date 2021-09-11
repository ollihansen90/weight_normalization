import torch
import matplotlib.pyplot as plt
import os

prepath = "auswertungen/"
#path = prepath+"auswertung.pt"
path = prepath+"auswertung.pt"
data = torch.load(path)

colorlist = 10*["b"]+10*["r"]+10*["k"]
accliste_train = data["accliste_train"].cpu()
accliste_test = data["accliste_test"].cpu()

n_epochs = 20
MLP_BN_data = torch.load("MLP_BN/MLP_BN_100_mean.pt")

plt.figure()
for ii, col in enumerate(colorlist):
    plt.plot(accliste_train[ii,:], col, alpha=0.3)
plt.plot(MLP_BN_data["accliste_train"][:n_epochs].cpu(), "m")
plt.grid()
plt.savefig("accliste_train.png")

plt.figure()
for ii, col in enumerate(colorlist):
    plt.plot(accliste_test[ii,:], col, alpha=0.3)
plt.plot(MLP_BN_data["accliste_test"][:n_epochs].cpu(), "m")  
plt.grid()
plt.savefig("accliste_test.png")

"""for kk in range(3):
    plt.figure()
    for ii, col in enumerate(colorlist):
        plt.plot(accliste_train[kk*15+ii,:], col, alpha=0.3)
    for jj in range(5):
        plt.plot(accliste_train[-jj,:], "y", alpha=0.3)
    plt.grid()
    plt.savefig("accliste_train"+str(kk))

    plt.figure()
    for ii, col in enumerate(colorlist):
        plt.plot(accliste_test[kk*15+ii,:], col, alpha=0.3)
    for jj in range(5):
        plt.plot(accliste_train[-jj,:], "y", alpha=0.3)
    plt.grid()
    plt.savefig("accliste_test"+str(kk))"""
"""losses = torch.tensor([[0,0]])
for file in os.listdir(prepath):
    if "lossliste" in file:
        #print(file)
        try:
            #print(file)
            data = torch.load(prepath+file).cpu()
            #print(data.shape)
        except IndexError:
            continue
        if data.shape[1]<100:
            addthis = torch.cat((data[:,[0]], data[:,[-1]]), -1)
            #print(addthis)
            losses = torch.cat((losses, addthis), 0)
print(losses.shape)
#data = torch.load(path)

mininorms = data["mininorms"]
maxinorms = data["maxinorms"]

plt.figure()
for i in range(4):
    plt.plot(mininorms[1:,i])
plt.legend(["Layer {}".format(i) for i in range(4)])
plt.title("Minima in Norms")
plt.show()
plt.savefig("mininorms.png")

plt.figure()
for i in range(4):
    plt.plot(maxinorms[1:,i])
plt.legend(["Layer {}".format(i) for i in range(4)])
plt.title("Maxima in Norms")
plt.show()
plt.savefig("maxinorms.png")

#lossliste = data["lossliste"].cpu()
#print(lossliste[:,0], lossliste[:,-1], sep="\n")
losses = losses[losses[:,0]<1e5,:]
plt.figure()
plt.scatter(losses[:,0], losses[:,-1])
plt.plot([0,1500], [0,1500])
plt.savefig("hier.png")"""


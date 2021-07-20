import torch
import matplotlib.pyplot as plt

prepath = "auswertungen/"
path = prepath+"auswertung.pt"
data = torch.load(path)
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
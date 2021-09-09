import torch

modelstamp = [  "1629137985", 
                "1629136348", 
                "1629135939", 
                "1629135530", 
                "1629135121", 
                "1629134712", 
                "1629134301", 
                "1629133892", 
                "1629133483", 
                "1629133074", 
                "1629132665", 
                "1629132255",
                "1629112006",
                "1629111590",
                "1629111174",
                "1629110754",
                "1629110334",
                "1628859996"
            ]
torch.set_printoptions(edgeitems=10_000)
mittelwert1 = 0
mittelwert2 = 0
with open("parameterlist.txt", "w") as file:
    for stamp in modelstamp:
        print(stamp)
        model = torch.load("models/model_"+stamp+".pt")
        for idx, param in enumerate(model.named_parameters()):
            #print(ii, param[1].shape)
            #idx = int(param[0].split(".")[2])
            if idx%4==2 or idx%4==3:
                #file.write(str(param)+"\n")
                print(idx, "\t", torch.mean(param[1]).item(), "\t", torch.std(param[1]).item())
                if idx==10:
                    mittelwert1 += torch.mean(param[1]).item()
                if idx==11:
                    mittelwert2 += torch.mean(param[1]).item()
print(mittelwert1/len(modelstamp))
print(mittelwert2/len(modelstamp))
import wandb
import os
from dotenv import load_dotenv

load_dotenv()
wand_key = os.getenv("WANDB_KEY")
if not wand_key:
    raise ValueError("Nenhuma API_KEY encontrada. Verifique o arquivo .env.")
print(f"A chave do wandB foi carregada com sucesso!")

wandb.login(wand_key)

#coloque aqui os valores dos seu hiper parametros
cnn_type = "" #coloque o nome da cnn que for fazer (vgg16 para a vgg16 pura,vgg16-tl para vgg16 com transfer learning, resnet50 para resnet50 densenet121 para a DenseNet121)
batch = 0
epochs = 0
learning_rate = 0
frozen_layers = 0

if cnn_type == "vgg16":
    run = wandb.init(
        entity="macuco-vinicius-ufjf",
        project="VC-Trabalho-Pratico",
        config={
            "architecture": "VGG16",
            "dataset": "CIFAR-100",
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch": batch
        }
    )
    # coloque aqui a sua lógica para treinar a rede

    run.log({"accuracy": 0, "loss": 0}) #torca aqui pelo sua mAP e loss
    run.finish()
else:
    run = wandb.init(
        entity="macuco-vinicius-ufjf",
        project="VC-Trabalho-Pratico",
        config={
            "architecture": cnn_type,
            "dataset": "CIFAR-100",
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch": batch,
            "frozen_layers": frozen_layers
        }
    )
    # aqui vem a lógica do treinamento da imageNet

    if(cnn_type == "vgg16-tl"):
        #lógica da vgg16 com o transfer learning

        #exportando os dados para o wandb
        run.log({"accuracy": 0, "loss": 0})
        run.finish()

    elif(cnn_type == "resnet50"):
        #lógica da resNet50

        # exportando os dados para o wandb
        run.log({"accuracy": 0, "loss": 0})
        run.finish()

    elif(cnn_type == "densenet121"):
        #lógica da denseNet121

        # exportando os dados para o wandb
        run.log({"accuracy": 0, "loss": 0})
        run.finish()

    else:
        #Nessa caso estamos salvando os dados da imageNet
        run.log({"accuracy": 0, "loss": 0})
        run.finish()






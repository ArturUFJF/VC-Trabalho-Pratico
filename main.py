import wandb
import os
from dotenv import load_dotenv
from datetime import datetime
from notion_client import Client


load_dotenv()
wandb.login(key=os.getenv("WANDB_KEY"))
notion_key = os.getenv("NOTION_KEY")
notion_database = os.getenv("NOTION_DATABASE")

if notion_key and notion_database:
    notion = Client(auth=notion_key)
    print("✅ Notion conectado.")
else:
    notion = None
    print("⚠️ Notion não configurado (verifique o .env).")

def send_to_notion(run_object, architecture, accuracy=0, loss=0,status="Finished"):
    if not notion:
        return
    try:
        notion.pages.create(
            parent={"database_id": notion_database},
            properties={
                "Name": {
                    "title": [{"text": {"content": run_object.name}}]
                },
                "Architecture": {
                    [{"text": {"name": architecture}}]
                },
                "WandBLink": {
                    "url":run_object.url
                },
                "Accuracy":{
                    "number":accuracy
                },
                "Loss":{
                    "number":loss
                },
                "Status":{
                    [{"select":{"name": status}}]
                }
            }
        )
    except Exception as e:
        print(f"❌ Erro ao salvar no Notion: {e}")


def main():
    #Define qual a rede a ser treinada ("vgg16" para a vgg16 da Lívia, "vgg16-tl" para a "vgg16" do Artur, "resnet50-tl" ou "densenet121-tl")
    #Qualquer outro valor aqui treinara apenas a ImageNet.
    cnn_type = "vgg16"

    # Hiperparâmetros
    config = {
        "architecture": cnn_type,
        "dataset": "CIFAR-100",
        "epochs": 5,
        "batch_size": 32,
        "learning_rate": 0.001,
        "frozen_layers": 5 if "tl" in cnn_type else 0
    }

    # 1. INICIA A RUN, preparando para armazenar dados na wandB
    run = wandb.init(
        project="VC-Trabalho-Pratico",
        entity="macuco-vinicius-ufjf",
        name=f"{cnn_type}-run-{datetime.now().strftime('%H%M')}",
        config=config,
        reinit=True  # Importante se for rodar várias seguidas no mesmo script
    )

    try:
        metrics = {}

        # 2. CHAMA A FUNÇÃO EM OUTRO ARQUIVO
        if cnn_type == "vgg16":
            #Lógica pra a VGG16 da Lívia
            print(f"{cnn_type}-run-{datetime.now().strftime('%H%M')} metrics:")


        else:
            #Treina a ImageNet para o Transfer Learning
            print(f"Treinando a ImageNet para o Transfer Learning")


            if cnn_type == "vgg16-tl":
                #Lógica para a VGG16 do Artur
                print(f"{cnn_type}-run-{datetime.now().strftime('%H%M')} metrics:")

            elif cnn_type == "resnet50-tl":
                #Lógica para a ResNet50
                print(f"{cnn_type}-run-{datetime.now().strftime('%H%M')} metrics:")

            elif cnn_type == "densenet121-tl":
                #Lógica para a DenseNet121
                print(f"{cnn_type}-run-{datetime.now().strftime('%H%M')} metrics:")

            else:
                print(f"Nenhuma rede foi escolhida para o Transfer Learning, somente a ImageNet rodou")

        send_to_notion(run, cnn_type, accuracy, loss)#precisamos salvar a a acurácia e a loss finais dos treinos com esses nomes.

    except Exception as e:
        print(f"❌ Erro Crítico durante a execução: {e}")
        wandb.alert(title="Falha no Treino", text=str(e))
        send_to_notion(run, cnn_type, status="Failed")

    finally:
        #Fecha a wandB para a próxima run
        run.finish()


if __name__ == "__main__":
    main()






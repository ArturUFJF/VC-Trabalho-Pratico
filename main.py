import wandb
import os
from dotenv import load_dotenv
from datetime import datetime
from notion_client import Client
from Modelos.DenseNet121 import treinar_densenet_tl
from pathlib import Path

directory = Path(__file__).resolve().parent
env = directory / '.env'


load_dotenv(dotenv_path=env)
wand_key = os.getenv("WANDB_KEY")
wandb.login(key=wand_key.strip(), relogin=True)
notion_key = os.getenv("NOTION_KEY")
notion_database = os.getenv("NOTION_DATABASE")



if notion_key and notion_database:
    notion = Client(auth=notion_key.strip())
    print("✅ Notion conectado.")
else:
    notion = None
    print("⚠️ Notion não configurado (verifique o .env).")

def send_to_notion(run_object, architecture, accuracy=0, loss=0, status="Finished"):
    if not notion:
        return
    try:
        notion.pages.create(
            parent={"database_id": notion_database.strip()},
            properties={
                "Runs": {
                    "title": [{"text": {"content": run_object.name}}]
                },

                "Architecture": {
                    "rich_text": [{"text": {"content": architecture}}]
                },

                "WandBLink": {
                    "url": run_object.url
                },
                "Accuracy": {
                    "number": float(accuracy)
                },
                "Loss": {
                    "number": float(loss)
                },
                "Status": {
                    "select": {"name": status}
                }
            }
        )
        print("✅ Dados salvos no Notion com sucesso!")
    except Exception as e:
        print(f"❌ Erro ao salvar no Notion: {e}")


def main():
    #Define qual a rede a ser treinada ("vgg16" para a vgg16 da Lívia, "vgg16-tl" para a "vgg16" do Artur, "resnet50-tl" ou "densenet121-tl")
    #Qualquer outro valor aqui e nehuma rede é escolhida.
    cnn_type = "densenet121-tl"

    # Hiperparâmetros, a gente muda tudo por aqui
    config = {
        "architecture": cnn_type,
        "dataset": "ImageNet + CIFAR-10" if "tl" in cnn_type else "CIFAR-10",
        "epochs": 5,
        "batch_size": 128,
        "learning_rate": 0.001,
        "unfrozen_layers": 0 if "tl" in cnn_type else 0
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


        elif cnn_type == "vgg16-tl":
            #Lógica para a VGG16 do Artur
            print(f"{cnn_type}-run-{datetime.now().strftime('%H%M')} metrics:")

        elif cnn_type == "resnet50-tl":
            #Lógica para a ResNet50
            print(f"{cnn_type}-run-{datetime.now().strftime('%H%M')} metrics:")

        elif cnn_type == "densenet121-tl":
            #Lógica para a DenseNet121
            print(f"{cnn_type}-run-{datetime.now().strftime('%H%M')} metrics:")
            metrics = treinar_densenet_tl(config)

        else:
            print(f"Nenhuma rede foi escolhida")

        send_to_notion(run, cnn_type, metrics['accuracy'], metrics['loss'])#precisamos salvar a a acurácia e a loss finais dos treinos com esses nomes.

    except Exception as e:
        print(f"❌ Erro Crítico durante a execução: {e}")
        wandb.alert(title="Falha no Treino", text=str(e))
        send_to_notion(run, cnn_type, status="Failed")

    finally:
        #Fecha a wandB para a próxima run
        run.finish()


if __name__ == "__main__":
    main()






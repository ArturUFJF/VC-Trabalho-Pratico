import wandb
import os
from dotenv import load_dotenv
from datetime import datetime
from Modelos.DenseNet121 import trainer_densenet_tl
from Modelos.VGG16 import trainer_vgg16
from Notion.notion_exporter import send_to_notion
from sklearn.metrics import classification_report
from pathlib import Path

directory = Path(__file__).resolve().parent
env = directory / '.env'
load_dotenv(dotenv_path=env)
wand_key = os.getenv("WANDB_KEY")
wandb.login(key=wand_key.strip(), relogin=True)

def main():
    #Define qual a rede a ser treinada (
    # "vgg16" para a vgg16 da Lívia, "vgg16-tl" para a "vgg16" do Artur, "resnet50-tl" ou "densenet121-tl")
    #Qualquer outro valor aqui e nehuma rede é escolhida.
    cnn_type = "vgg16"

    # Hiperparâmetros, a gente muda tudo por aqui
    # EXPERIMENTO 2: Menos épocas (19) - ponto ótimo observado no Exp1
    config = {
        "architecture": cnn_type,
        "dataset": "ImageNet + CIFAR-10" if "tl" in cnn_type else "CIFAR-10",
        "epochs": 19,  # MUDOU: 20 -> 19 (Experimento 2 - melhor epoch do Exp1)
        "batch_size": 128,
        "learning_rate": 0.01,
    }

    # 1. INICIA A RUN, preparando para armazenar dados na wandB
    run = wandb.init(
        project="VC-Trabalho-Pratico",
        entity="macuco-vinicius-ufjf",
        name=f"{cnn_type}-run-{datetime.now().strftime('%H%M')}",
        config=config,
    )

    try:
        metrics = {}

        # 2. CHAMA A FUNÇÃO EM OUTRO ARQUIVO
        if cnn_type == "vgg16":
            #Lógica pra a VGG16 da Lívia
            print(f"{cnn_type}-run-{datetime.now().strftime('%H%M')} metrics:")
            metrics = trainer_vgg16(config)


        elif cnn_type == "vgg16-tl":
            #Lógica para a VGG16 do Artur
            print(f"{cnn_type}-run-{datetime.now().strftime('%H%M')} metrics:")

        elif cnn_type == "resnet50-tl":
            #Lógica para a ResNet50
            print(f"{cnn_type}-run-{datetime.now().strftime('%H%M')} metrics:")

        elif cnn_type == "densenet121-tl":
            #Lógica para a DenseNet121
            print(f"{cnn_type}-run-{datetime.now().strftime('%H%M')} metrics:")
            metrics = trainer_densenet_tl(config)

        else:
            print(f"Nenhuma rede foi escolhida")

        acc = metrics.get('accuracy',0)
        loss = metrics.get('loss',0)
        send_to_notion(run, cnn_type, acc, loss)#precisamos salvar a a acurácia e a loss finais dos treinos com esses nomes.

    except Exception as e:
        print(f"❌ Erro Crítico durante a execução: {e}")
        wandb.alert(title="Falha no Treino", text=str(e))
        send_to_notion(run, cnn_type, status="Failed")

    finally:
        #Fecha a wandB para a próxima run
        run.finish()


if __name__ == "__main__":
    main()






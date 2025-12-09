import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from wandb.integration.keras import WandbMetricsLogger
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import wandb

def treinar_densenet_tl(config):
    print(f"--- Iniciando o transfer learning da DenseNet121 ---")

    #Carrega os dados do dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

    #Ajusta as imagens para o padrão da DenseNEt
    x_train = preprocess_input(x_train)
    x_test = preprocess_input(x_test)

    #Encoding das classes do CIFAR-10
    y_train = to_categorical(y_train,100)
    y_test = to_categorical(y_test,100)

    #Carrega a DenseNet121 treinada pela ImageNet
    # include_top=False: Remove a camada final de 1000 classes
    # input_shape=(32, 32, 3): Define o tamanho das imagens do CIFAR
    base_model = DenseNet121(
        include_top=False,
        weights='imagenet',
        input_shape=(32, 32, 3),
    )

    base_model.trainable = False

    #Descongela Camadas
    for layer in base_model.layers[-config['unfrozen_layers']:]: layer.trainable = True

    model = Sequential([
        base_model,
        # O GlobalAveragePooling transforma os mapas de características 3D em um vetor 1D
        GlobalAveragePooling2D(),
        # saída de acordo com o CIFAR-10
        Dense(100, activation='relu'),
        ]
    )

    #Compilando
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    callbacks_list = [
        WandbMetricsLogger(log_freq='epoch')
    ]

    history = model.fit(
        x_train, y_train,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        validation_data=(x_test, y_test),
        callbacks=callbacks_list,
        verbose=1
    )

    #Gerando Matriz de Confusão

    #Obtem probabilidades e classes da rede
    y_pred_probs = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    try:
        wandb.log({
            "conf-matrix_interactive": wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_true_classes,
                preds=y_pred_classes,
                title='Matriz de Confusão',
            )
        })
    except Exception as e:
        print(f"Aviso: Não foi possível gerar Matrix Nativa do WandB: {e}")

    #Salva uma imagem estática só por garantia
    print(f"Gerando Matrix de Confusão local")
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    fig, ax = plt.subplots(figsize=(24, 24))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=ax
    )
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predito')
    plt.title('Matriz de Confusão - CIFAR-100')
    wandb.log({"conf_mat_image": wandb.Image(fig)})
    plt.close(fig)
    print("Imagem da Matriz enviada para o WandB.")


    final_loss = history.history['val_loss'][-1]
    final_accuracy = history.history['val_accuracy'][-1]

    print(f"--- Treino da DenseNet121 finalizado ---")
    print(f"Final loss: {final_loss}")
    print(f"Final accuracy: {final_accuracy}")

    return {"loss": final_loss, "accuracy": final_accuracy}






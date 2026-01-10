import tensorflow as tf
import tensorflow_datasets as tfds
from keras import mixed_precision
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Resizing, Lambda
from keras.utils import to_categorical
from wandb.integration.keras import WandbMetricsLogger
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import wandb

def get_cifar10_labels():
    # Serve apenas para carregar os nomes das classes do CIFAR-10.
    # TFDS pode exigir download/Internet; mantemos fallback para evitar falhas.
    try:
        builder = tfds.builder("cifar10")
        builder.download_and_prepare()
        return builder.info.features['label'].names
    except Exception:
        return [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]

def trainer_vgg16_tl(config):
    print(f"--- Iniciando o transfer learning da VGG16 ---")
    mixed_precision.set_global_policy("mixed_float16")
    print("Mixed precision policy:", mixed_precision.global_policy())

    try:
        class_names = get_cifar10_labels()
    except Exception as e:
        print(f"Erro ao carregar as labels via TFSD: {e}")
        return {}

    #Carrega os dados do dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    #Ajusta as imagens para o padrão da VGG16
    x_train = preprocess_input(x_train)
    x_test = preprocess_input(x_test)

    #Encoding das classes do CIFAR-10
    y_train = to_categorical(y_train,10)
    y_test = to_categorical(y_test,10)
  

    #Carrega a VGG16 treinada pela ImageNet
    # include_top=False: Remove a camada final de 1000 classes
    # input_shape=(224, 224, 3): Define o tamanho das imagens do CIFAR
    base_model = VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False

    #Descongela Camadas
    if config['unfrozen_layers'] > 0:
        for layer in base_model.layers[-config['unfrozen_layers']:]:
            layer.trainable = True

    model = Sequential([
        Resizing(224, 224, interpolation="bilinear", input_shape=(32, 32, 3)),
        Lambda(preprocess_input),
        base_model,
        # O GlobalAveragePooling transforma os mapas de características 3D em um vetor 1D
        GlobalAveragePooling2D(),
        # saída de acordo com o CIFAR-10
        Dense(10, activation='softmax'),
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

    final_val_acc = history.history['val_accuracy'][-1]
    wandb.run.summary["best_val_accuracy"] = final_val_acc

    #Mariz de Confusão
    # Obtem probabilidades e classes da rede
    y_pred_probs = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    #Gera uma matriz de confusão interativa no WandB
    try:
        wandb.log({
            f"conf-matrix_interactive_{config['architecture']}": wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_true_classes,
                preds=y_pred_classes,
                class_names=class_names,
                title=f"Matriz de Confusão -{config['architecture']}- CIFAR-10",
            )
        })
    except Exception as e:
        print(f"Aviso: Não foi possível gerar Matrix Nativa do WandB: {e}")

    #Figura Estática
    cm = confusion_matrix(y_true_classes, y_pred_classes)

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    plt.xticks(rotation=45)
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predito')
    plt.title('Matriz de Confusão -VGG16- CIFAR-10')
    plt.tight_layout()

    wandb.log({"conf_mat_image": wandb.Image(fig)})
    plt.close(fig)

    report = classification_report(y_true_classes, y_pred_classes, target_names=class_names, output_dict=True)
    class_metrics = {}
    for class_name in class_names:
        class_metrics[f"class_acc/{class_name}"] = report[class_name]['f1-score']

    #Enviando Imagem para comparação
    # idx = random.randint(0, len(x_test) - 1)
    # sample_image = x_test_raw[idx]
    #
    # real_class = y_true_classes[idx]
    # predicted_class = y_pred_classes[idx]
    #
    # real_class_name = class_names[real_class]
    # predicted_class_name = class_names[predicted_class]
    #
    # print(f"Enviando exemplo de validação")
    # wandb.log({
    #     "validaton.exemple": wandb.Image(
    #         sample_image,
    #         caption=f"index: {idx} | Real: {real_class_name} | Predicted: {predicted_class_name}"
    #     )
    # })

    final_loss = history.history['val_loss'][-1]
    final_accuracy = history.history['val_accuracy'][-1]

    wandb.log({
        "final_val_accuracy": final_accuracy,
        "final_val_loss": final_loss,
        **class_metrics,
    })

    return {"loss": final_loss, "accuracy": final_accuracy}






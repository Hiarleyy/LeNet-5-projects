#%%
import tensorflow as tf
# importação do modelo sequencial
from tensorflow.keras.models import Sequential
#camadas de classificação e filtragem
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
# importação do dataset
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
#%%
## Carregando o dataset do Mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()

## Pré processamento dos dados
# Redimensiona as imagens para incluir o canal de cor(1 para imagens cinzas)
x_train = x_train.reshape(-1,28,28,1).astype('float32')/255.0
x_test = x_test.reshape(-1,28,28,1).astype('float32')/255.0

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test,num_classes=10)


## Modelo sequencial do LeNet5
model = Sequential([
    Conv2D(filters=6, kernel_size=(5, 5), activation='tanh', input_shape=(28, 28, 1), padding='same'),
    AveragePooling2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=16, kernel_size=(5, 5), activation='tanh'),
    AveragePooling2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(units=120, activation='tanh'),
    Dense(units=84, activation='tanh'),
        Dense(units=10, activation='softmax')  # Camada de saída para classificação em 10 classes
    ])

## Compilação do modelo 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

## otimizador = Adam | função de perda = categorical_crossentropy | métrica = acurácia

## Treinamento do modelo
model.fit(x_train, y_train, batch_size=32, epochs=10, validation_split=0.2)

## Avaliação do modelo
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Acurácia no conjunto de teste: {test_acc:.2f}")


# %%
def mostrar_exemplos(numero, dataset, labels):
    if numero < 0 or numero > 9:
        print("Por favor, insira um número entre 0 e 9.")
        return

    # Converter rótulos one-hot para valores escalares
    labels_scalar = np.argmax(labels, axis=1)
    # Filtrar imagens correspondentes ao número
    indices = np.where(labels_scalar == numero)[0]
    print(f"Encontrados {len(labels)} exemplos para o número {numero}.")  # Mensagem de depuração
    if len(indices) == 0:
        print(f"Nenhum exemplo encontrado para o número {numero}.")
        return
    
    # Selecionar a primeira imagem para exibição
    exemplo = dataset[indices[:5]].reshape(-1,28, 28)  # Redimensionar para 2D para exibição
  
    # Plotar a imagem
    plt.figure(figsize=(10, 2))
    for i, exemplo in enumerate(exemplo):
        plt.subplot(1, 5, i + 1)
        plt.imshow(exemplo, cmap='gray')
        plt.axis('off')
    plt.suptitle(f"Exemplos do número {numero}", fontsize=16)
    plt.show()
# Chamadas da função ajustadas
while True:
    try:
        numero = int(input("Digite um número entre 0 e 9 para ver exemplos (ou -1 para sair): "))
        if numero == -1:
            print("Encerrando o programa.")
            break
        mostrar_exemplos(numero, x_train, y_train)
    except ValueError:
        print("Entrada inválida. Por favor, insira um número entre 0 e 9.")

# %%

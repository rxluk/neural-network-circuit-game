# Modulo NN 03: Tipos de redes neurais

Nem todo problema pede o mesmo tipo de rede.

## 1. Feedforward / MLP

Rede de vetor para vetor, sem memoria explicita.

E o tipo usado no projeto.

## 2. CNN

Rede especializada em imagens e padroes espaciais.

Se o carro usasse camera em vez de sensores numericos, CNN seria candidata forte.

## 3. RNN, LSTM e GRU

Redes para sequencias e dependencia temporal.

Fazem sentido quando o historico recente importa muito.

## 4. Transformer

Arquitetura baseada em atencao, muito forte em linguagem e outras modalidades.

Nao e necessaria para entender nem reconstruir este projeto.

## 5. Autoencoder, GAN e GNN

Valem conhecer em nivel panoramico, mas nao sao o centro do que voce precisa agora.

## 6. Como escolher o tipo certo

Pergunta principal:

- qual e a estrutura da entrada?

Se a entrada e:

- vetor pequeno: MLP
- imagem: CNN
- sequencia: RNN/LSTM/GRU/Transformer
- grafo: GNN

## 7. Por que o projeto escolheu bem

Aqui a entrada ja vem condensada em sensores. Logo, uma MLP pequena resolve bem o problema sem exagero de complexidade.

## Exercicios

### Exercicio 1

Escolha um problema de imagem, um de sequencia e um de controle simples. Diga qual tipo de rede faria mais sentido para cada um.

### Exercicio 2

Imagine uma versao futura deste projeto usando camera. Escreva como a entrada, a rede e o custo computacional mudariam.

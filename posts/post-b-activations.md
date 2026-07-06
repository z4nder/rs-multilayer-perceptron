# Por que camadas lineares sozinhas não funcionam e o que a ReLU resolve

<p align="center">
  <img src="" alt="banner do projeto" width="1000" />
</p>

No [post anterior](https://dev.to/z4nder/de-um-neuronio-para-uma-rede-matrizes-camadas-e-o-forward-pass-1ai8) montamos o forward pass com duas camadas lineares em sequência. A rede parece ter mais "profundidade" mas matematicamente ela não tem. Neste post vamos entender por que e o que a função de ativação resolve.

### Conteúdo

- 1 [Prólogo](#1)
- 2 [O colapso linear](#2)
- 3 [O que é uma função de ativação](#3)
- 4 [ReLU no gráfico](#4)
- 5 [ReLU no código](#5)
- 6 [Onde a ativação entra — e onde não entra](#6)
- 7 [Conclusão](#7)

---

### 1. Prólogo <a name="1"></a>

Já montamos o `forward pass` na nossa layer com duas camadas lineares em sequência

```rust
let z1 = camada1.forward(&dataset.inputs); // X @ W1 + b1
let z2 = camada2.forward(&z1);             // z1 @ W2 + b2
```

Parece que a rede tem mais "profundidade" agora mas matematicamente estamos apenas empilhando camadas lineares sem ganho real para o aprendizado. As coisas pareciam conceitualmente simples, mas precisamos de um pouco mais de teoria para chegar num resultado melhor. Não se desespere como eu fiz: mesmo que esses conceitos não fiquem totalmente claros de primeira, acredito que consegui chegar numa explicação que melhora bastante a intuição sobre o assunto.

---

### 2. O colapso linear <a name="2"></a>

Depois de construir a `Layer`, a ideia natural é empilhar várias, uma camada processa a entrada e passa para a próxima, que refina o resultado.

O problema é que duas camadas lineares em sequência sempre **colapsam numa única camada linear**.

Suponha 2 inputs e duas camadas de pesos.

```text
X = [10.0, 80.0]

W1 = [
 [ 0.5, -0.4]
 [ 0.2, -0.1]
]

b1 = [1.0, 2.0]
```

Na camada 1 aplicando o `forward` chegamos em `Z = X @ W1 + b1`

```text
z1 = 10×0.5  + 80×0.2  + 1 =  22
z2 = 10×(-0.4) + 80×(-0.1) + 2 = -10

Z = [22, -10]
```

Agora passamos `Z` para a camada 2.

```text
W2 = [
 [0.4, 0.2]
 [0.6, 0.1]
]

b2 = [0.5, 1.0]
```

Aplicando novamente o `forward`:

```text
Y = Z @ W2 + b2

y1 = 22×0.4 + (-10)×0.6 + 0.5 = 3.3
y2 = 22×0.2 + (-10)×0.1 + 1.0 = 4.4

Y = [3.3, 4.4]
```

Apesar de termos usado duas camadas tudo isso ainda é equivalente a uma única transformação linear(Y = X @ W + b)

```
X
 ↓
Layer(2,2)
 ↓
Layer(2,2)
 ↓
Y

OU para 1 unico neurônio que é o método `y = x * w + b;`

z = x * w1 + b1;
y = z * w2 + b2;
```

É como se o treinamento tivesse aprendido um único conjunto de pesos equivalente ao efeito combinado dos dois.

Isso acontece porque multiplicações e somas são operações lineares.

Se colocarmos uma ativação no meio:

```
X
 ↓
Linear
 ↓
ReLU
 ↓
Linear
 ↓
Y

OU

z = x*w1 + b1
a = ReLU(z)
y = a*w2 + b2
```

O ponto principal é que **não existe mais um único `w` e `b` que reproduzam exatamente esse comportamento para todos os valores de x.**

Por isso costumamos dizer:

Sem ativação, várias camadas são apenas uma regressão linear disfarçada.
Com ativação, a rede passa a representar funções muito mais complexas.

---

### 3. O que é uma função de ativação <a name="3"></a>

Uma função de ativação é aplicada **elemento a elemento** à saída de uma camada antes de passar para a próxima.

O papel dela é introduzir **não-linearidade**, um comportamento que nenhuma combinação de pesos e bias consegue reproduzir.

A mais simples é a **ReLU**:

```
ReLU(x) = max(0, x)
```

Ela zera os negativos e deixa os positivos passarem:

```text
entrada: [-3.0,  2.0, -0.5,  1.8]
saída:   [ 0.0,  2.0,  0.0,  1.8]
```

**Por que zerar negativos remove a linearidade?**

Uma função é linear se `f(a + b) = f(a) + f(b)`. Testando com ReLU:

```text
ReLU(-1) + ReLU(1) = 0 + 1 = 1
ReLU(-1 + 1) = ReLU(0) = 0

1 ≠ 0
```

A propriedade não vale. O "zerar" cria um comportamento assimétrico onde positivos passam, negativos não e isso quebra o que permite o colapso.

Voltando ao exemplo do tópico anterior e agora com ReLU entre as camadas:

```text
Z = [22, -10]   ← saída da camada 1

ReLU([22, -10]) = [22, 0]   ← o -10 foi zerado

Y = [22, 0] @ W2 + b2

y1 = 22×0.4 + 0×0.6 + 0.5 = 9.3
y2 = 22×0.2 + 0×0.1 + 1.0 = 5.4

Y = [9.3, 5.4]
```

Sem ReLU o resultado era `[3.3, 4.4]`. Com ReLU é `[9.3, 5.4]`. O neurônio 2 foi zerado no meio do caminho — e agora **não existe nenhum `W` e `b` que, numa única camada, produza `[9.3, 5.4]` para esse input e ao mesmo tempo se comporte diferente para outros inputs onde o neurônio 2 não seria zerado.** O comportamento depende do sinal dos valores intermediários, e isso quebra o colapso.

---

### 4. ReLU no gráfico <a name="4"></a>

A forma mais direta de entender o que a ativação faz é olhando o gráfico de duas redes com a mesma arquitetura `(1 → 4 → 1)` e os mesmos pesos — a única diferença é a ReLU entre as camadas.

![Sem ativação vs Com ReLU](../../../outputs/03_activations.png)

À esquerda: **linha reta**, sempre — independente de quantas layers.

À direita: segmentos retos com **dobras**. Cada neurônio oculto contribui com um ponto de dobra onde seu valor pré-ativação cruza zero. Com 4 neurônios ocultos, até 4 dobras possíveis.

Essas dobras são o que permite à rede aproximar padrões complexos que uma linha reta nunca conseguiria representar.

---

### 5. ReLU no código <a name="5"></a>

A implementação em Rust é trivial:

```rust
pub fn relu(input: &Matrix) -> Matrix {
    let data: Vec<f64> = (0..input.rows)
        .flat_map(|i| (0..input.cols).map(move |j| (i, j)))
        .map(|(i, j)| input.get(i, j).max(0.0))
        .collect();

    Matrix::new(input.rows, input.cols, data)
}
```

Recebe uma `Matrix`, devolve uma `Matrix` do mesmo formato com os negativos zerados. O pipeline fica:

```rust
let z1 = camada1.forward(&input); // X @ W1 + b1
let a1 = relu(&z1);               // zera os negativos
let z2 = camada2.forward(&a1);    // a1 @ W2 + b2
```

---

### 6. Onde a ativação entra — e onde não entra <a name="6"></a>

A ativação é aplicada **depois** do forward de cada camada **oculta**. A camada de saída não leva ativação.

```
input
  ↓
Layer 1 → forward → ReLU   ← camadas ocultas levam ativação
  ↓
Layer 2 → forward           ← saída final, sem ativação
  ↓
previsão
```

Se zerássemos os negativos na saída, a rede nunca conseguiria prever valores negativos o que seria um problema para qualquer tarefa de regressão.

---

### 7. Conclusão <a name="7"></a>

Neste post entendemos por que empilhar camadas lineares não adiciona expressividade e como a ReLU resolve isso com uma operação simples.

O que foi construído:

- A função `relu`
- Pipeline completo: `Layer1 → ReLU → Layer2`
- Visualização do colapso linear vs ReLU no gráfico

No próximo post: **backpropagation** como medir o erro e fazer a rede aprender de verdade.

---

### Referências

- [Código-fonte do projeto](https://github.com/z4nder/rs-multilayer-perceptron)
- [Neural Network from Scratch — vídeo que inspirou essa série](https://www.youtube.com/watch?v=GkiITbgu0V0&t=477s)
- [Post anterior — Matrizes, camadas e forward pass](https://dev.to/z4nder/ia-do-zero-de-um-neuronio-para-uma-rede-matrizes-camadas-e-o-forward-pass)

---

Se este post fizer sentido pra você, o próximo passo é ensinar a rede a aprender com os próprios erros isso vem no próximo post da série.

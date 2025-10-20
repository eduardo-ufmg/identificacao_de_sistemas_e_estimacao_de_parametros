#### **Exercício 1: Fundamentos dos Estimadores Recursivos**

O objetivo aqui é comparar o desempenho (viés e variância) de diferentes famílias de estimadores recursivos para um sistema com parâmetros constantes.

**1.a: Estimadores Não-Polarizados (Tipo Gradiente e Gauss-Newton)**

* **Observado:**
    O método de Gradiente obteve menor variância e viés. Para estimação de a1, o viés não foi desprezível, mas é baixo. Para b1, é despresível. GN teve variância tão alta que a média foi deslocada por outliers.

**1.b: Estimador de Aproximação Estocástica**

* **Observado:**
    O viés foi aumentado para o método de gradiente. O viés foi fortemente reduzido para o método GN. Em ambos, a variância foi fortemente reduzida.

**1.c: Estimador de Mínima Covariância**

* **Observado:**
    Viés desprezível e variância baixíssima.

**1.d: Mínimos Quadrados Recursivo (MQR / RLS)**

* **Observado:**
    Viés desprezível e variância baixíssima.

**1.e: Comparação Geral e Impacto do Número de Amostras (N)**

* **Observado:**
    * Viés cai consistentemente para Gradiente, Estocastica de Gradiente, Mínima Covariância e MQR. Para GN, não há tendencia bem definida. Para Estocástica de GN, há queda no viés, mas um tanto inconsistente, principalmente nas dezenas de amostras.

    * Variância cai consistentemente para MQR e Mínima Covariância. Para Gradiente, aumenta e estabiliza. Para Estocástica do Gradiente, aumenta e, a partir das centenas de amostras, cai. Para GN, cai até as dezenas e estabiliza. Para Estocástica de GN, aumenta nas dezenas e, a partir das centenas, cai.

**1.f: Trajetória dos Parâmetros**

* **Observado:**
    * Gradiente aproxima rápido, mas flutua bastante e é ruidozo.
    * GN é extremamente ruidozo e não tende a valor algum.
    * Estocástica de Gradiente aproxima bem lentamente, mas com pouco ruído.
    * Estocástica de GN ainda é bastante ruidozo, mas menos do que o original e flutua em torno do valor real.
    * Mínima Covariância e MQR aproximam rápido e ficam estáveis.

#### **Exercício 2: Parâmetros Variantes no Tempo**

O objetivo é avaliar a capacidade de um estimador de rastrear parâmetros que mudam com o tempo.

**2.a: Efeito do Fator de Esquecimento ($\lambda$)**

* **Observado:**
    Lambdas altos deixam a estimativa suave, mas lenta. Lambdas altos deixam-na rápida, mas ruidoza. O melhor foi 0,95. Ainda bem ruidozo, mas consistentemente próximo do real.

**2.b: Efeito da Persistência de Excitação**

* **Observado:**
    O aumento no hold aumenta muito o atraso e um pouco o ruído das estimativas.

#### **Exercício 3: Condições Não Ideais**

O objetivo é testar a robustez dos estimadores quando as premissas sobre o ruído são violadas.

**3.a: Ruído com Média Não-Nula**

* **Observado:**
    * O viés para a1 é aumentado para todos os estimadores. Para Gradiente e GN, a variância também aumentou.
    * Lambdas maiores levam a estimativas deslocadas para a1. Lambdas menores conseguem acompanhar os parametros, mas com mais ruído. Para b1, o ruído aumenta.

**3.b: Ruído Colorido**

* **Observado:**
    O viés é aumentado e mais difícil de corrigir, em todos os casos.

#### **Exercício 4: Aplicação em Dados Reais**

O objetivo é aplicar os métodos a um problema prático e interpretar os resultados.

**4.b e 4.d: Comparação entre MQR e MQR-FF**

* **Observado:**
    MQR converge e estabiliza para A. Parece haver pequena adaptação para B. MQR-FF varia pouco, mas consistentemente, para A e B. Difícil concluir se é ruído ou acompanhamento dos parametros.

**4.c: Estimador de Espaço de Estados (Filtro de Kalman)**

* **Observado:**
    A fica mais suave, parecendo mais um acompanhamento dos parâmetros reais. B parece ficar mais ruidoso, mas com tendencias claras.


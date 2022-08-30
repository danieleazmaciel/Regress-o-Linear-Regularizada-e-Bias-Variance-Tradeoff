import sys
import numpy as np
from scipy import optimize
from matplotlib import pyplot

def trainLinearReg(linearRegCostFunction, X, y, lambda_=0.0, maxiter=200):
    """
    Treina uma regressão linear utilizando a função optimize.minimize do scipy.

    Parâmetros
    ----------
    X : tipo_array
        O conjunto de dados no shape (m x n+1). O termo de viés já deve ter sido concatenado.

    y : tipo_Array
        Valor alvo da função para cada ponto de dado. Um vetor na forma (m,).

    lambda_ : float, opcional
        O parâmetro de regularização.

    maxiter : int, apcional
        Número máximo de iterações para o algoritmo de otimização.

    Retorna
    -------
    theta : tipo_array
        Os parâmetros para a regressão linear. Este é um vetor na forma (n+1,).
    """
    # Inicializa Theta
    initial_theta = np.zeros(X.shape[1])

    # Cria uma "abreaviação" para a função custo a ser minimizada
    costFunction = lambda t: linearRegCostFunction(X, y, t, lambda_)

    # Agora, a função custo é uma função que assume apenas um argumento
    options = {'maxiter': maxiter}

    # Minimiza utilizando o scipy
    res = optimize.minimize(costFunction, initial_theta, jac=True, method='TNC', options=options)
    return res.x


def featureNormalize(X):
    """
    Normaliza os features em X retornando uma versão normalizada de X onde o valor médio de cada
    feature é 0 e seu desvio padrão é 1. Este é usualmente um bom passo pré-processamento a ser 
    feito quando trabalhando com algoritmos de aprendizado de máquina.

    Parâmetros
    ----------
    X : tipo_array
        Um conjunto de dados o qual é uma matriz (M x n), onde m é o número de 
        amostras e n é o número de features de cada amostra.

    Retorna
    -------
    X_norm : tipo_array
        Cnojunto de dado de input normalizado

    mu : tipo_array
        Um vetor de tamanho n correspondendo ao valor médio de cada feature sobre todos as amostras.

    sigma : tipo_array
        Um vetor de tamanho n correspondendo ao desvio padrão para cada feature sobre todas as
        amostras.
    """
    mu = np.mean(X, axis=0)
    X_norm = X - mu

    sigma = np.std(X_norm, axis=0, ddof=1)
    X_norm /= sigma
    return X_norm, mu, sigma


def plotFit(polyFeatures, min_x, max_x, mu, sigma, theta, p):
    """
    Plota um ajuste polinomial sobre uma figura existente.
    Também funciona com uma regressão linear.
    Plota o ajuste polinomial aprendido com potênci p e feature normalization (mu, sugma).

    Parâmetros
    ----------
    polyFeatures : função
        Uma função que gera features polinomiais a partir de um único feature.

    min_x : float
        O valor mínimo do feature.

    max_x : float
        O valor máximo do feature.

    mu : float
        O valor médio do feature sobre o conjunto de treino.

    sigma : float
        O desvio padrão do feature sobre o conjunto de treino.

    theta : tipo_array
        O parâmetros da regressão polinomial treinada.

    p : int
        Ordem polinomial.
    """
    # Plotamos um intervalo um pouco superior aos valores min e max para termo
    # uma ideia de como o ajuste varia fora do intervalo definido pelo conjunto
    # pontos do dado
    x = np.arange(min_x - 15, max_x + 25, 0.05).reshape(-1, 1)

    # Mapeia os valores de X
    X_poly = polyFeatures(x, p)
    X_poly -= mu
    X_poly /= sigma

    # Adiciona os termos de interceptação
    X_poly = np.concatenate([np.ones((x.shape[0], 1)), X_poly], axis=1)

    # Plota
    pyplot.plot(x, np.dot(X_poly, theta), '--', lw=2)

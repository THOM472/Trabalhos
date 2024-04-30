import pandas as pd
import numpy as np
import matplotlib.pyplot as fig


d=pd.read_excel(r"C:\Users\thoma\Documents\baseaps.xlsx", sheet_name='Aba 2', usecols='A:G', index_col=0, 
                nrows=120, skiprows=range(1, 3))

desvio = d.std().tolist()
tot = d.columns.tolist()

retorno = d.mean().tolist()

#Tratamento de carteiras aleatórias e alocação de espaços
n = 1000000

cart = np.random.random((n, len(tot)))
cart = cart/np.sum(cart, keepdims = True, axis=1)


#Índice de sharpe

retcart = np.sum(cart * np.array(retorno), axis=1)
riscocart = np.sqrt(np.sum((cart * np.array(desvio))**2, axis=1))

melhor_cart = np.argmax((retcart - retorno[0]) / riscocart)
melhor_ret = retcart[melhor_cart]
melhor_risco = riscocart[melhor_cart]


#W, Ativo livre de risco e sharpe

W = cart[melhor_cart]
retrf = 0.0097192
riscorf = 0


sharpe = (retcart - retrf) / riscocart

melhor_aloc = np.argmax(sharpe)
melhor_ret = retcart[melhor_aloc]
melhor_risco = riscocart[melhor_aloc]

#gráfico
fig.figure(figsize=(10,8))

fig.scatter(riscocart, retcart, c=sharpe, cmap="viridis")
fig.ylabel('Retorno')
fig.xlabel('Risco')
fig.title('Fronteira eficiente carteira')
fig.scatter(riscorf, retrf, color ='red')
fig.scatter(melhor_risco, melhor_ret, color='c', marker = 'd' , s = 180) #melhor carteira
fig.plot([riscorf, melhor_risco], [retrf, melhor_ret], color='orange')

fig.show()

print ("Pesos da carteira ótima")
for i,peso in enumerate (W): {
        print(f"Ativo{i+1}: {peso: 4f}")}

#Calculo de carteira para cada perfil

A1 = 1

u1 = retcart - 0.5*A1*(riscocart**2)
maxu1 = np.argmax(u1)
maxuret1 = retcart[maxu1]
maxurisc1 = riscocart [maxu1]
pesorisco1 = (melhor_ret - retrf)/(A1*melhor_risco**2)
pesorf1 = 1 - pesorisco1

retfin1 = (pesorisco1*maxuret1)+(pesorf1 * retrf)
riscofin1 = (pesorisco1*maxurisc1)+(pesorf1*riscorf)

#Gráfico do perfil
fig.scatter(riscofin1, retfin1, color='lime', marker = "8", s=80)


A2 = 3

u2 = retcart - 0.5*A2*(riscocart**2)
maxu2 = np.argmax(u2)
maxuret2 = retcart[maxu2]
maxurisc2 = riscocart [maxu2]
pesorisco2 = (melhor_ret - retrf)/(A2*melhor_risco**2)
pesorf2 = 1 - pesorisco2

retfin2 = (pesorisco2*maxuret2)+(pesorf2 * retrf)
riscofin2 = (pesorisco2*maxurisc2)+(pesorf2*riscorf)

#Gráfico do perfil
fig.scatter(riscofin2, retfin2, color='purple', marker = "8", s=80)


A3 = 5

u3 = retcart - 0.5*A3*(riscocart**2)
maxu3 = np.argmax(u3)
maxuret3 = retcart[maxu3]
maxurisc3 = riscocart [maxu3]
pesorisco3 = (melhor_ret - retrf)/(A3*melhor_risco**2)
pesorf3 = 1 - pesorisco3

retfin3 = (pesorisco3*maxuret3)+(pesorf3 * retrf)
riscofin3 = (pesorisco3*maxurisc3)+(pesorf3*riscorf)

#Gráfico do perfil
fig.scatter(riscofin3, retfin3, color='k', marker = "8", s=80)


#finalizando gráficos
fig.tight_layout()
fig.show()



print(f"Risco na alocação (A=1): {maxurisc1}")
print(f"Peso risco ótimo (A=1): {pesorisco1}")
print(f"Peso no rf (A=1): {pesorf1}")
print(f"Retorno final (A=1): {retfin1}")
print(f"Risco na alocação (A=3): {maxurisc2}")
print(f"Peso risco ótimo (A=3): {pesorisco2}")
print(f"Peso no rf (A=3): {pesorf2}")
print(f"Retorno final (A=3): {retfin2}")
print(f"Risco na alocação (A=5): {maxurisc3}")
print(f"Peso risco ótimo (A=5): {pesorisco3}")
print(f"Peso no rf (A=5): {pesorf3}")
print(f"Retorno final (A=5): {retfin3}")



import numpy as np
import pandas as pd

from sklearn import preprocessing as pp
from sklearn.svm import SVC

import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------------------------
# ----- VAMOS A CARGAR Y PREPROCESAR UN POQUITO LOS DATOS -------------------------------------
#----------------------------------------------------------------------------------------------

# Datos a emplear: German Credit
data=pd.read_csv('C:/Users/Agus/Google Drive/Universidad/5 - Quinto/TFG/Datasets/german.data', sep=' ',
                 header=None)

names = ['AcSt','Dur','CrHis','Pur','CrAm','Sav','Empl','InR','PS','DG','Res','Prop','Age','oIP','Hou','exC','Job','Lia','Tel','For','Target']
data.columns = names

# Son ordinales:
iOrd = [0,2,5,6,11,16]
nO = len(iOrd)
namesOrd = np.array(names)[iOrd]
# Son categóricos:
iCat = [3,8,9,13,14,18,19]
namesCat = np.array(names)[iCat]
# Son enteros:
iInt = [1,10,12,15,17]
nZ = len(iInt)
namesInt = np.array(names)[iInt]
# Son continuos:
iCon = [4,7]
nR = len(iCon)
namesCon = np.array(names)[iCon]

# Codificación de los datos ordinales a enteros ---------------------------------------------

dataOrd = data.iloc[:,iOrd].copy()

dictOrd = [['A11','A14','A12','A13', 'nada'],
           ['A30','A31','A32','A33','A34'],
           ['A65','A61','A62','A63','A64'],
           ['A71','A72','A73','A74','A75'],
           ['A121','A122','A123','A124','nada'],
           ['A171','A172','A173','A174','nada']]

matDicOrd = np.array(dictOrd).transpose()

ordEncoder = pp.OrdinalEncoder()
ordEncoder.fit(matDicOrd)
ordEncoder.feature_names_in_ = ['AcSt','CrHis','Sav','Empl','Prop','Job']
ordEncoder.feature_names_in_

Xord = pd.DataFrame(ordEncoder.transform(dataOrd))
Xord.columns = ordEncoder.feature_names_in_

# Codificación de los datos categóricos a binarios ------------------------------------------

# Sin dropear
dataCat = data.iloc[:,iCat].copy()

catEncoder = pp.OneHotEncoder()
catEncoder.sparse = False
catEncoder.drop = None
Xcat = pd.DataFrame(catEncoder.fit_transform(dataCat))
Xcat.columns = Xcat.columns.astype('str')
# Distribución de etiquetas: (dropeando)
# 3:{0-8}, 8:{9-11}, 9:{12-13}, 13:{14:15}, 14:{16-17}, 18:{18}, 19:{19}
# Distribución de etiquetas: (sin dropear)
# 3:{0-9}, 8:{10-13}, 9:{14-16}, 13:{17:19}, 14:{20-22}, 18:{23,24}, 19:{25,26}
nC = Xcat.shape[1]

# Enteros -----------------------------------------------------------------------------------
Xint = data.iloc[:,iInt].copy()

# Continuos ------------------------------------------------------------------------------
Xcon = data.iloc[:,iCon].copy()


# Vector objetivo ---------------------------------------------------------------------------
y = data.iloc[:,20].astype('category')

# Vamos a hacer un decodificador también.-----------------------------------------

n1 = nR; n2= n1 + nZ; n3 = n2 + nO; n4 = n3 + nC;


def decode(x):
    x = np.array(x,ndmin=2)
    xcon = np.array(x[0,0:n1],ndmin=2)
    xint = np.array(x[0,n1:n2],ndmin=2)
    xord = np.array(x[0,n2:n3],ndmin=2)
    xord = ordEncoder.inverse_transform(xord)
    xcat = np.array(x[0,n3:n4],ndmin=2)
    xcat = catEncoder.inverse_transform(xcat)
    return pd.DataFrame(np.hstack([xcon,xint,xord,xcat]))



# --------------------------------------------------------------------------------------------
# -- UNA VEZ LOS DATOS ESTÁN CODIFICADOS, VAMOS A ENTRENAR -----------------------------------
# --------------------------------------------------------------------------------------------

X = pd.concat([Xcon, Xint, Xord, Xcat],axis=1) #3 + 4 + 6 + 27

scaler = pp.StandardScaler()
X_st = pd.DataFrame(scaler.fit_transform(X))
media = scaler.mean_
varianza = scaler.var_
desv_est = np.sqrt(varianza)


predictor = SVC()
predictor.class_weight='balanced'#{1:1,2:5}
predictor.kernel='linear'
predictor.C = 0.1
predictor.fit(X_st,y)

# Vamos a ver la eficacia al clasificar los propios datos de entrenamiento
if False:

    predicciones=predictor.predict(X_st)
    aciertos = predicciones == y
    plt.close("all")
    pd.crosstab(y,aciertos).plot(kind='bar',stacked=True)
    plt.title('Correct predictions across target label')
    plt.xlabel('Target label')
    plt.show()
    plt.savefig('training data efficiency')
    pd.crosstab(y,predicciones)
    # Acierto de cerca del 75% en ambos casos

#MESSIRVE

# --------------------------------------------------------------------------------------
# -------- SACAMOS EL HIPERPLANO SEPARADOR DEL PREDICTOR -------------------------------
# --------------------------------------------------------------------------------------

coef_st = sum(predictor.coef_)
bias_st = predictor.intercept_
# Este es para trabajar con las variables estandarizadas; hay que "desestandarizarlo"
coef = coef_st/desv_est
np.savetxt('coefbalaced.txt',coef)
#coef = [1,1]
offset = media*coef_st/desv_est
bias = bias_st - sum(offset)
bias = float(bias)
# El hiperplano ya estaría
f = lambda x : np.dot(x,coef) + bias

# Comprobación: CORRECTA
a = (f(X.iloc[:,:]) > 0)
b = predicciones == 2
a==b

#Ojo que la categoría deseable es la 1, que corresponde a f < 0

# Para tenerlo más mascado, vamos a separar los coeficientes según tipo de variable
coefCont = coef[0:n1]
coefInt = coef[n1:n2]
coefOrd = coef[n2:n3]
coefCat = coef[n3:n4]

# --------------------------------------------------------------------------------------
# -------- PREPARAMOS LOS PROBLEMAS DE OPTIMIZACIÓN ------------------------------------
# --------------------------------------------------------------------------------------

from gurobipy import *

def SCOP(x0,w_r,w_z,a0,a1,a2,bigM):

    # Segmentamos el vector de decisión

    r0 = x0[0,0:n1]
    z0 = x0[0,n1:n2]
    o0 = x0[0,n2:n3]
    c0 = x0[0,n3:n4]

    # Manipulamos los pesos que vamos a emplear para las distancias ponderadas

    w = np.append(w_r,np.append(w_z,np.ones(n4-n2)))
    W_r = np.diag(w_r*w_r)
    W_z = np.diag(w_z*w_z)

    # Iniciamos el modelo

    m = Model('Counterfactual')

    # Añadimos las variables

    r = m.addMVar(nR,name="x",vtype=GRB.CONTINUOUS,lb = np.zeros(nR))
    z = m.addMVar(nZ,name="z",vtype=GRB.INTEGER,lb = np.zeros(nZ))
    o = m.addMVar(nO,name="o",vtype=GRB.INTEGER,lb = np.zeros(nO),ub = np.array([3,3,4,4,3,3]))
    c = m.addMVar(nC,name="c",vtype=GRB.BINARY)
    t = m.addMVar(n4,name="t",vtype=GRB.CONTINUOUS)
    s = m.addMVar(n4,name="s",vtype=GRB.BINARY)

    #Añadimos las restricciones para las categóricas
    # 3:{0-9}, 8:{10-13}, 9:{14-16}, 13:{17:19}, 14:{20-22}, 18:{23,24}, 19:{25,26}
    A = np.zeros((4,nC))
    A[0,14:17] = np.ones((1,3))
    A[1,17:20] = np.ones((1,3))
    A[2,20:23] = np.ones((1,3))
    A[3,23:25] = np.ones((1,2))

    b = np.ones(4)

    m.addMConstr(A=A,x=c,sense='=',b=b,name='CategoricalConstraint')

    # De inmutabilidad, a ver cómo funcionan aquí dentro:
    Aim = np.zeros((16,nC))
    Aim[0:10,0:10] = np.eye(10)
    Aim[10:14,10:14] = np.eye(4)
    Aim[14:16,25:27] = np.eye(2)

    bim = Aim @ c0

    m.addMConstr(A=Aim,x=c,sense='=',b=bim,name='ImmutabilityConstraint')

    # Añadimos la restricción de clase
    def clase(rr,zz,oo,cc):
        return rr @ coefCont + oo @ coefOrd + zz @ coefInt + cc @ coefCat + bias

    m.addConstr(clase(r,z,o,c) <= 0, "Validity")

    # Añadimos las restricciones para la L1
    T1 = np.eye(nR,40,k=0)
    T2 = np.eye(nZ,40,k=3)
    T3 = np.eye(nO,40,k=7)
    T4 = np.eye(nC,40,k=13)

    m.addConstr(r-r0 <= T1 @ t,name='L1_Cons_cont_1')
    m.addConstr(r-r0 >= -T1 @ t,name='L1_Cons_cont_2')
    m.addConstr(z-z0 <= T2 @ t,name='L1_Cons_int_1')
    m.addConstr(z-z0 >= -T2 @ t,name='L1_Cons_int_2')
    m.addConstr(o-o0 <= T3 @ t,name='L1_Cons_ord_1')
    m.addConstr(o-o0 >= -T3 @ t,name='L1_Cons_ord_2')
    m.addConstr(c-c0 <= T4 @ t,name='L1_Cons_cont_1')
    m.addConstr(c-c0 >= -T4 @ t,name='L1_Cons_cont_2')

    # Añadimos las restricciones para la L0
    m.addConstr(r-r0 <= T1 @ s * bigM,name='L0_Cons_cont_1')
    m.addConstr(r-r0 >= -T1 @ s * bigM,name='L0_Cons_cont_2')
    m.addConstr(z-z0 <= T2 @ s * bigM,name='L0_Cons_int_1')
    m.addConstr(z-z0 >= -T2 @ s * bigM,name='L0_Cons_int_2')
    m.addConstr(o-o0 <= T3 @ s * bigM,name='L0_Cons_ord_1')
    m.addConstr(o-o0 >= -T3 @ s * bigM,name='L0_Cons_ord_2')
    m.addConstr(c-c0 <= T4 @ s * bigM,name='L0_Cons_cont_1')
    m.addConstr(c-c0 >= -T4 @ s * bigM,name='L0_Cons_cont_2')


    # Añadimos los objetivos
    def distanciaL2(rr,zz,oo,cc):
        disr = rr @ W_r @ rr - 2 * r0 @ W_r @ rr
        disz = zz @ W_z @ zz - 2 * z0 @ W_z @ zz
        diso = oo@oo - 2 * o0 @ oo
        disc = cc@cc - 2 * c0 @ cc
        return disr + disz + diso + disc

    #def distanciaL1(rr,zz,oo,cc):
    def distanciaL1(tt):
        return w @ tt

    #def distanciaL0(rr,zz,oo,cc):
    ws = np.append(np.ones(n3),0.5*np.ones(nC))

    def distanciaL0(ss):
        return ws @ ss

    m.setObjective(a0*distanciaL0(s) + a1*distanciaL1(t) + a2*distanciaL2(r,z,o,c), GRB.MINIMIZE)
    #m.setObjective(distanciaL0(s), GRB.MINIMIZE)

    m.update()

    # Resolvemos

    m.optimize()

    sol = np.array(m.x,ndmin=2)
    x = sol[0,0:n4]
    t = sol[0,n4:2*n4]
    s = sol[0,2*n4:3*n4]
    L0 = distanciaL0(s)
    L1 = distanciaL1(t)
    L2 = distanciaL2(x[0:n1],x[n1:n2],x[n2:n3],x[n3:n4])

    return({'x':x,'t':t,'s':s,'L0':L0,'L1':L1,'L2':L2})

def MCOP(x0,w_r,w_z,a0,a1,a2,bigM,b1):

    # Segmentamos el vector de decisión

    r0 = x0[0,0:n1]
    z0 = x0[0,n1:n2]
    o0 = x0[0,n2:n3]
    c0 = x0[0,n3:n4]

    # Manipulamos los pesos que vamos a emplear para las distancias ponderadas

    w = np.append(w_r,np.append(w_z,np.ones(n4-n2)))
    W_r = np.diag(w_r*w_r)
    W_z = np.diag(w_z*w_z)

    # Iniciamos el modelo

    m = Model('Counterfactual')

    # Añadimos las variables

    r = m.addMVar(nR,name="x",vtype=GRB.CONTINUOUS,lb = np.zeros(nR))
    z = m.addMVar(nZ,name="z",vtype=GRB.INTEGER,lb = np.zeros(nZ))
    o = m.addMVar(nO,name="o",vtype=GRB.INTEGER,lb = np.zeros(nO),ub = np.array([3,3,4,4,3,3]))
    c = m.addMVar(nC,name="c",vtype=GRB.BINARY)
    t = m.addMVar(n4,name="t",vtype=GRB.CONTINUOUS)
    s = m.addMVar(n4,name="s",vtype=GRB.BINARY)

    #Añadimos las restricciones para las categóricas
    # 3:{0-9}, 8:{10-13}, 9:{14-16}, 13:{17:19}, 14:{20-22}, 18:{23,24}, 19:{25,26}
    A = np.zeros((4,nC))
    A[0,14:17] = np.ones((1,3))
    A[1,17:20] = np.ones((1,3))
    A[2,20:23] = np.ones((1,3))
    A[3,23:25] = np.ones((1,2))

    b = np.ones(4)

    m.addMConstr(A=A,x=c,sense='=',b=b,name='CategoricalConstraint')

    # De inmutabilidad, a ver cómo funcionan aquí dentro:
    Aim = np.zeros((16,nC))
    Aim[0:10,0:10] = np.eye(10)
    Aim[10:14,10:14] = np.eye(4)
    Aim[14:16,25:27] = np.eye(2)

    bim = Aim @ c0

    m.addMConstr(A=Aim,x=c,sense='=',b=bim,name='ImmutabilityConstraint')

        # Añadimos las restricciones para la L1
    T1 = np.eye(nR,40,k=0)
    T2 = np.eye(nZ,40,k=3)
    T3 = np.eye(nO,40,k=7)
    T4 = np.eye(nC,40,k=13)

    m.addConstr(r-r0 <= T1 @ t,name='L1_Cons_cont_1')
    m.addConstr(r-r0 >= -T1 @ t,name='L1_Cons_cont_2')
    m.addConstr(z-z0 <= T2 @ t,name='L1_Cons_int_1')
    m.addConstr(z-z0 >= -T2 @ t,name='L1_Cons_int_2')
    m.addConstr(o-o0 <= T3 @ t,name='L1_Cons_ord_1')
    m.addConstr(o-o0 >= -T3 @ t,name='L1_Cons_ord_2')
    m.addConstr(c-c0 <= T4 @ t,name='L1_Cons_cont_1')
    m.addConstr(c-c0 >= -T4 @ t,name='L1_Cons_cont_2')

    # Añadimos las restricciones para la L0
    m.addConstr(r-r0 <= T1 @ s * bigM,name='L0_Cons_cont_1')
    m.addConstr(r-r0 >= -T1 @ s * bigM,name='L0_Cons_cont_2')
    m.addConstr(z-z0 <= T2 @ s * bigM,name='L0_Cons_int_1')
    m.addConstr(z-z0 >= -T2 @ s * bigM,name='L0_Cons_int_2')
    m.addConstr(o-o0 <= T3 @ s * bigM,name='L0_Cons_ord_1')
    m.addConstr(o-o0 >= -T3 @ s * bigM,name='L0_Cons_ord_2')
    m.addConstr(c-c0 <= T4 @ s * bigM,name='L0_Cons_cont_1')
    m.addConstr(c-c0 >= -T4 @ s * bigM,name='L0_Cons_cont_2')


    # Añadimos los objetivos
    def distanciaL2(rr,zz,oo,cc):
        disr = rr @ W_r @ rr - 2 * r0 @ W_r @ rr
        disz = zz @ W_z @ zz - 2 * z0 @ W_z @ zz
        diso = oo@oo - 2 * o0 @ oo
        disc = cc@cc - 2 * c0 @ cc
        return disr + disz + diso + disc

    #def distanciaL1(rr,zz,oo,cc):
    def distanciaL1(tt):
        return w @ tt

    #def distanciaL0(rr,zz,oo,cc):
    ws = np.append(np.ones(n3),0.5*np.ones(nC))

    def distanciaL0(ss):
        return ws @ ss

    # Objetivo de clase
    def clase(rr,zz,oo,cc):
        return rr @ coefCont + oo @ coefOrd + zz @ coefInt + cc @ coefCat + bias

    m.setObjective(a0*distanciaL0(s) + a1*distanciaL1(t) + a2*distanciaL2(r,z,o,c) + b1*clase(r,z,o,c), GRB.MINIMIZE)
    #m.setObjective(distanciaL0(s), GRB.MINIMIZE)

    m.update()

    # Resolvemos

    m.optimize()

    sol = np.array(m.x,ndmin=2)
    x = sol[0,0:n4]
    t = sol[0,n4:2*n4]
    s = sol[0,2*n4:3*n4]
    L0 = distanciaL0(s)
    L1 = distanciaL1(t)
    L2 = distanciaL2(x[0:n1],x[n1:n2],x[n2:n3],x[n3:n4])
    F = clase(x[0:n1],x[n1:n2],x[n2:n3],x[n3:n4])

    return({'x':x,'t':t,'s':s,'L0':L0,'L1':L1,'L2':L2,'f':F})

#-----------------------------------------------------------------------------------

#Consideramos los pesos que vamos a aplicar a continuas y enteras:
#sus desviaciones tipicas a la -1.

w_r = 1/desv_est[0:n1]
w_z = 1/desv_est[n1:n2]


for i in range(100):
    if y[i]==2:
        x0 = np.array(X.iloc[i,:],ndmin=2)

        resultados = pd.DataFrame()
        vecF = pd.DataFrame()

        for j in range(30):
            ctr = MCOP(x0=x0,w_r=w_r,w_z=w_z,a0=1,a1=1,a2=2,bigM = 1e15,b1=j)
        #    ctr['s']
        #    ctr['f']
        #    1/(1+np.exp(ctr['f']))
            filaF = pd.DataFrame([ctr['f'],1/(1+np.exp(ctr['f']))]).transpose()
            vecF = vecF.append(filaF)
            resultados = resultados.append(decode(ctr['x']))

        resultados.index=range(30)
        resultados.columns= np.concatenate([namesCon,namesInt,namesOrd,namesCat])
        resultados = resultados.astype('float64',errors='ignore')

        vecF.columns = ['f','prob']
        vecF.index=range(30)


        resultadosFinal = pd.concat([resultados,vecF],axis=1)
        resultadosFinal = resultadosFinal.drop_duplicates()

        for k in range(n1):
            resultadosFinal[namesCon[k]] = pd.to_numeric(resultadosFinal[namesCon[k]])
        for k in range(nZ):
            resultadosFinal[namesInt[k]] = pd.to_numeric(resultadosFinal[namesInt[k]])


        archivo = 'res_x' + str(i) +'.csv'

        resultadosFinal.to_csv(archivo, sep=';', decimal = ',')
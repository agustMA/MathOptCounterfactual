
import numpy as np
import pandas as pd

from sklearn import preprocessing as pp
from sklearn.svm import SVC

#import seaborn as sns
#import matplotlib.pyplot as plt

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

def decode2(xcon,xint,xord,xcat):
    xord = ordEncoder.inverse_transform(xord)
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
predictor.class_weight={1:1,2:5}
predictor.kernel='rbf'
predictor.C = 1e15
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

#MESSIRVE

# --------------------------------------------------------------------------------------
# -------- SACAMOS LOS PARÁMETROS INTERNOS DEL PREDICTOR --------------------------
# --------------------------------------------------------------------------------------

#Recordemos que es un rbf:
b0 = predictor._intercept_
support = predictor.support_vectors_
nSup = support.shape[0]
a = predictor.dual_coef_
gamma = predictor._gamma

def decision(x):
    return predictor.decision_function(np.array((x-media)/desv_est,ndmin=2))[0]

    # Comprobación: CORRECTA
#a = decision(X.iloc[:,:]) > 0
#b = predicciones == 2
#a==b

#Ojo que la categoría deseable es la 1, que corresponde a f < 0

# Para tenerlo más mascado, vamos a separar los coeficientes según tipo de variable

# --------------------------------------------------------------------------------------
# -------- PREPARAMOS LOS PROBLEMAS DE OPTIMIZACIÓN ------------------------------------
# --------------------------------------------------------------------------------------

from pyomo.environ import *
import os

os.environ['NEOS_EMAIL']='martin.aguera.agustin@gmail.com'

#SCOP(x0=x0,w_r=w_r,w_z=w_z,a0=1,a1=1,a2=2,bigM = 1e15,b1=j)

def SCOP(x0,a0,a1,a2,bigM):



    # ----------------- INICIAMOS EL MODELO ---------------------------------

    m = ConcreteModel()

    # ----------------- AÑADIMOS LOS SETS -----------------------------------

    m.cont = RangeSet(0,nR-1)
    m.int = RangeSet(0,nZ-1)
    m.ord = RangeSet(0,nO-1)
    m.cat = RangeSet(0,nC-1) 

    m.sup = RangeSet(0,nSup-1)

    m.total = RangeSet(0,n4-1)

    # --------------- AÑADIMOS LOS PARÁMETROS -------------------------------
    # La instancia original:
    m.r0 = Param(m.cont, initialize = x0[0,0:n1])
    m.z0 = Param(m.int, initialize = x0[0,n1:n2])
    m.o0 = Param(m.ord, initialize = x0[0,n2:n3])
    m.c0 = Param(m.cat, initialize = x0[0,n3:n4])

    # Los support vectors:
    def rj_init(model,j,i):
        return support[j,i]
    m.rj = Param(m.sup,m.cont, initialize = rj_init)
    def zj_init(model,j,i):
        return support[j,n1+i]
    m.zj = Param(m.sup,m.int, initialize = zj_init)
    def oj_init(model,j,i):
        return support[j,n2+i]
    m.oj = Param(m.sup,m.ord, initialize = oj_init)
    def cj_init(model,j,i):
        return support[j,n3+i]
    m.cj = Param(m.sup,m.cat, initialize = cj_init)

    # Los pesos de los support vectors:
    m.aj = Param(m.sup, initialize = a[0,:])

    # El resto de parámetros de la función de decisión:
    m.b0 = Param(initialize = b0[0])
    m.gamma = Param(initialize = gamma)

    # Las varianzas, porque recordemos que los datos entran al clasificador estandarizados
    m.var_r = Param(m.cont, initialize = varianza[0:n1])
    m.var_z = Param(m.int, initialize = varianza[n1:n2])
    m.var_o = Param(m.ord, initialize = varianza[n2:n3])
    m.var_c = Param(m.cat, initialize = varianza[n3:n4])
    m.std = Param(m.total, initialize = 1/desv_est)

    # ---------------- AÑADIMOS LAS VARIABLES -------------------------------

    m.r = Var(m.cont,domain=NonNegativeReals, initialize=m.r0)

    limitesZ = {0:0, 1:0, 2:m.z0[2], 3:0, 4:0}
    def fbZ(model,i):
        return (limitesZ[i],None)
    m.z = Var(m.int,domain=NonNegativeIntegers,bounds = fbZ, initialize=m.z0)

    limitesO = {0:3, 1:3, 2:4, 3:4, 4:3, 5:3}
    def fbO(model,i):
        return (0, limitesO[i])
    m.o = Var(m.ord,domain=NonNegativeIntegers,bounds = fbO, initialize=m.o0)

    m.c = Var(m.cat,domain=Binary, initialize=m.c0)

    m.t = Var(m.total, domain = NonNegativeReals, initialize = 0)

    m.s = Var(m.total, domain = Binary, initialize = 0)

    # ---------------- AÑADIMOS LAS RESTRICCIONES ---------------------------
    # La de validez
    def l2_comp_r_ker(model,j):
        return sum((model.r[i]-model.rj[j,i])**2/m.var_r[i] for i in model.cont)
    def l2_comp_z_ker(model,j):
        return sum((model.z[i]-model.zj[j,i])**2/m.var_z[i] for i in model.int)
    def l2_comp_o_ker(model,j):
        return sum((model.o[i]-model.oj[j,i])**2/m.var_o[i] for i in model.ord)
    def l2_comp_c_ker(model,j):
        return sum((model.c[i]-model.cj[j,i])**2/m.var_c[i] for i in model.cat)
    
    def l2_ker(model,j):
        sr = l2_comp_r_ker(model,j) 
        sz = l2_comp_z_ker(model, j)
        so = l2_comp_o_ker(model, j)
        sc = l2_comp_c_ker(model, j)
        return m.gamma*(sr+sz+so+sc)

    def ker(model,j):
        return exp(-l2_ker(model,j))

    def decision(model):
        return model.b0 + sum(ker(model,j)*model.aj[j] for j in model.sup)

    m.validity = Constraint(expr = (None, decision(m), 0))

    #Las de suma 1 para las categóricas
    def Cat2(model):
        return model.c[14] + model.c[15] + model.c[16] == 1
    m.consCat2 = Constraint(rule = Cat2)
    def Cat3(model):
        return model.c[17] + model.c[18] + model.c[19] == 1
    m.consCat3 = Constraint(rule = Cat3)
    def Cat4(model):
        return model.c[20] + model.c[21] + model.c[22] == 1
    m.consCat4 = Constraint(rule = Cat4)
    def Cat5(model):
        return model.c[23] + model.c[24] == 1
    m.consCat5 = Constraint(rule = Cat5)



    # 3:{0-9}, 8:{10-13}, 9:{14-16}, 13:{17:19}, 14:{20-22}, 18:{23,24}, 19:{25,26}

    # De inmutabilidad, a ver cómo funcionan aquí dentro:
    m.immLia = Constraint(expr = (m.z[4]-m.z0[4],0))
    m.immPur = ConstraintList()
    for i in range(10):
        m.immPur.add(expr = (m.c[i]-m.c0[i],0))
    m.immPS = ConstraintList()
    for i in range(10,14):
        m.immPS.add(expr = (m.c[i]-m.c0[i],0))
    m.immFor = ConstraintList()
    for i in range(25,27):
        m.immFor.add(expr = (m.c[i]-m.c0[i],0))

    # Restricciones para la L1
    m.L1cons = ConstraintList()
    for i in range(nR):
        m.L1cons.add(expr = (None,m.r[i]-m.r0[i]-m.t[i],0))
        m.L1cons.add(expr = (None,m.r0[i]-m.r[i]-m.t[i],0))
    for i in range(nZ):
        m.L1cons.add(expr = (None,m.z[i]-m.z0[i]-m.t[i+n1],0))
        m.L1cons.add(expr = (None,m.z0[i]-m.z[i]-m.t[i+n1],0))
    for i in range(nO):
        m.L1cons.add(expr = (None,m.o[i]-m.o0[i]-m.t[i+n2],0))
        m.L1cons.add(expr = (None,m.o0[i]-m.o[i]-m.t[i+n2],0))
    for i in range(nC):
        m.L1cons.add(expr = (None,m.c[i]-m.c0[i]-m.t[i+n3],0))
        m.L1cons.add(expr = (None,m.c0[i]-m.c[i]-m.t[i+n3],0))

    ## Añadimos las restricciones para la L0
    m.L0cons = ConstraintList()

    for i in range(nR):
        m.L0cons.add(expr = (None,m.r[i]-m.r0[i]-m.s[i]*bigM,0))
        m.L0cons.add(expr = (None,m.r0[i]-m.r[i]-m.s[i]*bigM,0))
    for i in range(nZ):
        m.L0cons.add(expr = (None,m.z[i]-m.z0[i]-m.s[i+n1]*bigM,0))
        m.L0cons.add(expr = (None,m.z0[i]-m.z[i]-m.s[i+n1]*bigM,0))
    for i in range(nO):
        m.L0cons.add(expr = (None,m.o[i]-m.o0[i]-m.s[i+n2]*bigM,0))
        m.L0cons.add(expr = (None,m.o0[i]-m.o[i]-m.s[i+n2]*bigM,0))
    for i in range(nC):
        m.L0cons.add(expr = (None,m.c[i]-m.c0[i]-m.s[i+n3]*bigM,0))
        m.L0cons.add(expr = (None,m.c0[i]-m.c[i]-m.s[i+n3]*bigM,0))

    # ---------------------- AÑADIMOS LOS OBJETIVOS ----------------------
    # distancia L2
    def l2_r_obj(model):
        return sum((model.r[i]-model.r0[i])**2/m.var_r[i]  for i in model.cont)
    def l2_z_obj(model):
        return sum((model.z[i]-model.z0[i])**2/m.var_z[i]  for i in model.int)
    def l2_o_obj(model):
        return sum((model.o[i]-model.o0[i])**2/m.var_o[i]  for i in model.ord)
    def l2_c_obj(model):
        return sum((model.c[i]-model.c0[i])**2/m.var_c[i]  for i in model.cat)
    
    def l2_obj(model):
        return(l2_r_obj(model) + l2_z_obj(model) + l2_o_obj(model) + l2_c_obj(model))

    # distancia L1

    def l1(model):
        return summation(model.std,model.t)

    # distancia L0
    def l0(model):
        return sum(model.s[i] for i in model.total)
    # esto no tiene en cuenta la doble cuenta

    # OBJETIVO TOTAL
    def objetivo(model):
        return a0 * l0(model) + a1 * l1(model) + a2 * l2_obj(model)

    m.obj = Objective(rule=objetivo)

    # ------------------------------ RESOLVEMOS ------------------------------
    
    opt = SolverManagerFactory('neos')
    opt.solve(m, solver='bonmin')


    r = np.array(list(m.r.extract_values().values()),ndmin=2)
    z = np.array(list(m.z.extract_values().values()),ndmin=2)
    o = np.array(list(m.o.extract_values().values()),ndmin=2)
    c = np.array(list(m.c.extract_values().values()),ndmin=2)
    t = np.array(list(m.t.extract_values().values()),ndmin=2)
    s = np.array(list(m.s.extract_values().values()),ndmin=2)

    return {'modelo':m, 'r':r, 'z':z, 'o':o, 'c':c, 't':t, 's':s}


x0 = np.array(X.iloc[4,:],ndmin=2)
ctr = SCOP(x0,a0=20,a1=1,a2=1,bigM=1e6)
decode2(ctr['r'],ctr['z'],ctr['o'],ctr['c'])

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


for i in [4]:
    x0 = np.array(X.iloc[i,:],ndmin=2)
    if decision(x0)>0:

        resultados = pd.DataFrame()
        vecF = pd.DataFrame()

        for j in range(1):
            ctr = MCOP(x0=x0,a0=20,a1=1,a2=2,bigM = 1e6)#,b1=j)
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

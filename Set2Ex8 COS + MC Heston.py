import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

r = 0
np.random.seed(39)
def CIR_Sample(NoOfPaths,kappa,gamma,vbar,s,t,v_s):
    delta = 4.0 *kappa*vbar/gamma/gamma
    c= 1.0/(4.0*kappa)*gamma*gamma*(1.0-np.exp(-kappa*(t-s)))
    kappaBar = 4.0*kappa*v_s*np.exp(-kappa*(t-s))/(gamma*gamma*(1.0-np.exp(-kappa*(t-s))))
    sample = c* np.random.noncentral_chisquare(delta,kappaBar,NoOfPaths)
    return  sample

### Monte Carlo functions ###
def Heston_paths(NoOfPaths,NoOfSteps, T, r, S_0, kappa, gamma, rho, vbar, v_0):    
    Z1 = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    W1 = np.zeros([NoOfPaths, NoOfSteps+1])
    V = np.zeros([NoOfPaths, NoOfSteps+1])
    X = np.zeros([NoOfPaths, NoOfSteps+1])
    V[:,0]=v_0
    X[:,0]=np.log(S_0)
    
    time = np.zeros([NoOfSteps+1])
        
    dt = T / float(NoOfSteps)
    for i in range(0,NoOfSteps):
        # making sure that samples from normal have mean 0 and variance 1
        if NoOfPaths > 1:
            Z1[:,i] = (Z1[:,i] - np.mean(Z1[:,i])) / np.std(Z1[:,i])
        W1[:,i+1] = W1[:,i] + np.power(dt, 0.5)*Z1[:,i]
        
        # Exact samles for the variance process
        V[:,i+1] = CIR_Sample(NoOfPaths,kappa,gamma,vbar,0,dt,V[:,i])
        k0 = (r -rho/gamma*kappa*vbar)*dt
        k1 = (rho*kappa/gamma -0.5)*dt - rho/gamma
        k2 = rho / gamma
        X[:,i+1] = X[:,i] + k0 + k1*V[:,i] + k2 *V[:,i+1] + np.sqrt((1.0-rho**2)*V[:,i])*(W1[:,i+1]-W1[:,i])
        time[i+1] = time[i] +dt
        
    #Compute exponent
    S = np.exp(X)
    paths = {"time":time,"S":S}
    return paths


def payoff(S,K,cp):
    #european call payoff
    if cp == "c" or cp == 0:
        H = np.array([S-K, 0])
        return np.max(H)
    #european put payoff
    if cp == "p" or cp == 1:
        H = np.array([K-S, 0])
        return np.max(H)
    
def MC_option_price(X,r,NoOfSamples,T,K,cp):
    H = np.zeros([NoOfSamples])
    for j in range(0, NoOfSamples):
        H[j] = np.exp(-r*T)*payoff(X[j,-1],K,cp)
     
    optionprice = np.mean(H)
    return {"price": optionprice, "sample payoffs": H}

### COS Method functions ###
def COS_Price_CP(cf,CP,S_0,tau,K,N,L,r):
    # reshape K to a column vector
    # K = np.array(K).reshape([len(K),1])
    
    #assigning i=sqrt(-1)
    i = np.complex(0.0,1.0) 
    
    x0 = np.log(S_0 / K)   
    
    # truncation domain
    a = 0.0 - L * np.sqrt(tau)
    b = 0.0 + L * np.sqrt(tau)
    
    # sumation from k = 0 to k=N-1
    k = np.linspace(0,N-1,N).reshape([N,1])  
    u = k * np.pi / (b - a);  

    # Determine coefficients for Put Prices  
    H_k = CallPutCoefficients(CP,a,b,k)
       
    mat = np.exp(i * np.outer((x0 - a) , u))

    temp = cf(u) * H_k 
    temp[0] = 0.5 * temp[0]    
    
    value = np.exp(-r * tau) * K * np.real(mat.dot(temp))
         
    return value[0]

def CallPutCoefficients(CP,a,b,k):
    if str(CP).lower()=="c" or str(CP).lower()=="1":                  
        c = 0.0
        d = b
        coef = Chi_Psi(a,b,c,d,k)
        Chi_k = coef["chi"]
        Psi_k = coef["psi"]
        if a < b and b < 0.0:
            H_k = np.zeros([len(k),1])
        else:
            H_k      = 2.0 / (b - a) * (Chi_k - Psi_k)  
        
    elif str(CP).lower()=="p" or str(CP).lower()=="-1":
        c = a
        d = 0.0
        coef = Chi_Psi(a,b,c,d,k)
        Chi_k = coef["chi"]
        Psi_k = coef["psi"]
        H_k      = 2.0 / (b - a) * (- Chi_k + Psi_k)               
    
    return H_k    

def Chi_Psi(a,b,c,d,k):
    psi = np.sin(k * np.pi * (d - a) / (b - a)) - np.sin(k * np.pi * (c - a)/(b - a))
    psi[1:] = psi[1:] * (b - a) / (k[1:] * np.pi)
    psi[0] = d - c
    
    chi = 1.0 / (1.0 + np.power((k * np.pi / (b - a)) , 2.0)) 
    expr1 = np.cos(k * np.pi * (d - a)/(b - a)) * np.exp(d)  - np.cos(k * np.pi 
                  * (c - a) / (b - a)) * np.exp(c)
    expr2 = k * np.pi / (b - a) * np.sin(k * np.pi * 
                        (d - a) / (b - a))   - k * np.pi / (b - a) * np.sin(k 
                        * np.pi * (c - a) / (b - a)) * np.exp(c)
    chi = chi * (expr1 + expr2)
    
    value = {"chi":chi,"psi":psi }
    return value

def B_Heston(u,tau):
    i = np.complex(0.0,1.0)
    return 0

def C_Heston(u,tau,kappa,gamma,vbar,rho):
    i = np.complex(0.0,1.0)
    D1 = np.sqrt((kappa - gamma*rho*i*u)**2 + (u**2 + i*u)*gamma**2)
    g  = (kappa - gamma*rho*i*u - D1)/(kappa - gamma*rho*i*u + D1)
    res = (1 - np.exp(-D1*tau))*(kappa - gamma*rho*i*u - D1)/ \
        (gamma**2 *(1 - g*np.exp(-D1*tau)))
    return res

def A_Heston(u,tau,kappa,gamma,vbar,rho,r):
    i = np.complex(0.0,1.0)
    D1 = np.sqrt((kappa - gamma*rho*i*u)**2 + (u**2 + i*u)*gamma**2)
    g  = (kappa - gamma*rho*i*u - D1)/(kappa - gamma*rho*i*u + D1)
    term1 = r*i*u*tau
    term2 = kappa*vbar*tau*(kappa - gamma*rho*i*u - D1)/(gamma**2)
    term3 = -2*kappa*vbar*np.log((1 - g*np.exp(-D1*tau))/(1 - g))/(gamma**2)
    res = term1 + term2 + term3
    return res
    
def cF_Heston(tau,kappa,gamma,vbar,S_0,v_0,rho,r):
    i = np.complex(0.0,1.0)
    A = lambda u: A_Heston(u,tau,kappa,gamma,vbar,rho,r)
    B = lambda u: B_Heston(u,tau)
    C = lambda u: C_Heston(u,tau,kappa,gamma,vbar,rho)
    
    charfunc = lambda u: np.exp(A(u) + C(u)*v_0)
    return charfunc    
    
def main_calc():
    K = 100 
    T = 1
    S_0 = 100
    v_0 = 0.0175
    kappa = 1.5768
    gamma = 0.5751
    vbar = 0.0398
    rho = -0.5711 
    cp = "p"
    
    # setting for Monte Carlo simulation
    NoOfPaths = 50000
    NoOfSteps = 1000
   
    # setting for the COS method 
    tau = T
    
    # range for the expansion points
    N = 2**12
    L = 15
         
    # characteristic function for Heston model
    cF = cF_Heston(tau,kappa,gamma,vbar,S_0,v_0,rho,r)
    
    # COS option price
    COS_optionprice = COS_Price_CP(cF,cp,S_0,tau,K,N,L,r)[0]
    
    
    S = Heston_paths(NoOfPaths, NoOfSteps, T, r, S_0, kappa, gamma, rho, vbar, v_0)["S"]
    
    MC_optionprice = MC_option_price(S,r,NoOfPaths,T,K,cp)["price"]
    payoffs = MC_option_price(S,r,NoOfPaths,T,K,cp)["sample payoffs"]
    alpha=0.95
    CI = st.t.interval(alpha, NoOfPaths-1, loc=MC_optionprice, scale=st.sem(payoffs))
    
    msg = "The value of a Heston {0} option based on Monte Carlo is {1:5f} with {2} confidence interval [{3:5f},{4:5f}]" \
          " and based on COS method is {5:5f}".format(cp,MC_optionprice,alpha,CI[0],CI[1],COS_optionprice)
    print(msg)
    # plt.figure(1)
    # plt.grid()
    # plt.xlabel("x")
    # plt.ylabel("$f_X(x)$")
    # plt.plot(x,f_Vasicek,'-r')
    # plt.legend(["Vasicek density"])
    # plt.title("COS recovered density of BSV model")
    
    # plt.plot(K, COS_optionprice, "-r")
    # plt.plot(K, np.transpose(MC_optionprice), "--b")
    # plt.grid()
    # plt.xlabel("Strike K")
    # plt.ylabel("Value")
    # plt.title("COS vs Monte Carlo Option price")
    # plt.legend(["COS method","Monte Carlo"])        
    
main_calc()
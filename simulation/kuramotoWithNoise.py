import numpy as np
import matplotlib.pyplot as plt
from math import pi

def calc_order_param(Nosc,theta):
    real_sum=0.0
    img_sum=0.0
    r=0.0
    psi=0.0
    for i in range(Nosc):
        real_sum = real_sum+np.cos(theta[i])
        img_sum = img_sum+np.sin(theta[i])
    real_sum=real_sum/Nosc
    img_sum=img_sum/Nosc
    r=np.sqrt(real_sum*real_sum+img_sum*img_sum)
    psi = np.arccos(real_sum/r)
    return r,psi

np.random.seed(1) 
Nosc = 3
#Ntime = 1000
dt =0.001
thetas = []
times = []
theta0 = np.random.uniform(0,2*pi,Nosc)
# omega = np.random.normal( loc = 1.0, scale = .01, size = Nosc )
omega = 2.0 * pi * np.ones(Nosc)
print (omega)
theta = theta0
thetas.append(theta)
times.append(0.0)
order_param_r=[]
order_param_psi=[]

# alpha = .50   # noise st
alpha = .001   # noise st
K = .10
# for ntime in range(1,Ntime):
ntime=0
while ( (theta[0] / (2*pi)) < 20 ):
    r, psi = calc_order_param(Nosc,theta)
    order_param_r.append(r)
    order_param_psi.append(psi)
    ntime = ntime + 1
    theta = theta + dt * ( omega + K * r * np.sin(psi-theta)) + alpha * ( np.random.normal( loc = 0.0, scale = np.sqrt(dt), size = Nosc ) )
    thetas.append(theta)
    times.append( dt * ntime )

np.savetxt("K=0.1_thetas", thetas)
np.savetxt("K=0.1_orderParam", order_param_r)
np.savetxt("K=0.1_times", times)

# thetas=np.transpose(thetas)
# for i in range(Nosc):
#     plt.plot(times,np.sin(thetas[i]))
# plt.plot(times[0:-1],order_param_r,label='order param')
# plt.legend()
#     # plt.plot(times[0:-1],np.sin(order_param_psi))
# plt.show()

exit()
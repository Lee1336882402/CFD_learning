import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gif
import time
from tqdm import tqdm

def TDMA(a,b,c,d):
    #a = Lower Diag, b = Main Diag, c = Upper Diag, d = solution vector
    #source code from 'https://www.coder.work/article/1937578'
    n = len(d)
    w= np.zeros(n-1,float)
    g= np.zeros(n, float)
    p = np.zeros(n,float)
    
    w[0] = c[0]/b[0]
    g[0] = d[0]/b[0]

    for i in range(1,n-1):
        w[i] = c[i]/(b[i] - a[i-1]*w[i-1])
    for i in range(1,n):
        g[i] = (d[i] - a[i-1]*g[i-1])/(b[i] - a[i-1]*w[i-1])
    p[n-1] = g[n-1]
    for i in range(n-1,0,-1):
        p[i-1] = g[i-1] - w[i-1]*p[i]
    return p

class FDM():

    def __init__(self, init_fuc:callable, lbc:callable, rbc:callable, method=None, sigma=0.1, M_x=300, gama=1):
        if method is None:
            print('Warning! No method is initialized. Default method is FTCS.')
            method = 'FTCS'
        #if method not in ['FTCS','BTCS']:
        #    raise TypeError
        self.method = method
        self.sigma = sigma
        self.M_x = M_x
        self.delta_x = 1/(M_x-1)
        self.delta_t = sigma*(self.delta_x)**2/gama
        self.all_data = np.zeros(M_x)
        self.now_step = 0
        self.init_fuc = init_fuc
        self.all_data = init_fuc(self.all_data)
        self.lbc = lbc
        self.rbc = rbc

    def forward(self, n_step):
        if self.method == 'FTCS':
            for i in range(n_step):
                self.now_step += 1
                temp_data = np.zeros_like(self.all_data)
                temp_data[0] = self.lbc(self.now_step*self.delta_t)
                temp_data[-1] = self.rbc(self.now_step*self.delta_t)
                for j in range(self.M_x):
                    if j==0 or j==self.M_x-1:continue
                    temp_data[j] = self.sigma*self.all_data[j+1] + (1-2*self.sigma)*self.all_data[j] + self.sigma*self.all_data[j-1]
                self.all_data = temp_data
        elif self.method == 'BTCS':
            #using TDMA
            for i in range(n_step):
                self.now_step += 1
                temp_data = np.zeros_like(self.all_data)
                solution_vector = -self.all_data[1:-1]
                solution_vector[0] -= self.sigma*self.lbc(self.now_step*self.delta_t)
                solution_vector[-1] -= self.sigma*self.rbc(self.now_step*self.delta_t)
                temp_data[1:-1] = TDMA(self.sigma*np.ones(self.M_x-3),(-1-2*self.sigma)*np.ones(self.M_x-2),self.sigma*np.ones(self.M_x-3),solution_vector)
                temp_data[0] = self.lbc(self.now_step*self.delta_t)
                temp_data[-1] = self.rbc(self.now_step*self.delta_t)
                self.all_data = temp_data
    def reinit(self):
        self.now_step = 0
        self.all_data = self.init_fuc(self.all_data)



    def postprocess(self,step_number):
        plt.figure(figsize=(18,8))
        for plotindex in range(6):
            self.forward(step_number[plotindex])
            plt.subplot(2,3,plotindex+1)
            plt.plot(np.arange(self.M_x)/(self.M_x-1),self.all_data)
            plt.xlim((0,1))
            plt.ylim((0,3))
            plt.text(0,1.75,self.method+',boundary condition %d, time step %f\n node number %d, plot at step %d'%(1,self.delta_t,self.M_x,self.now_step))
            plt.xlabel('x')
            plt.ylabel('u')
        plt.savefig('.\\save\\'+self.method+'_%d_%f.png'%(self.M_x,self.sigma))

    def gif_create(self, step_interval):
        # reference: https://blog.csdn.net/qingfengxd1/article/details/113725721
        gif.options.matplotlib["dpi"] = 500
        @gif.frame
        def plot():
            #plt.clf()
            plt.plot(np.arange(self.M_x)/(self.M_x-1),self.all_data)
            plt.text(0,1.75,self.method+',boundary condition %d, time step %f\n node number %d, plot at step %d'%(2,self.delta_t,self.M_x,self.now_step))
            plt.xlim((0,1))
            plt.ylim((0,3))
            plt.xlabel('x')
            plt.ylabel('u')
        frames = []
        start_time =time.time()
        for i in tqdm(range(30)):
            self.forward(step_interval)
            frame = plot()
            frames.append(frame)
        end_time = time.time()
        gif.save(frames, self.method+'_%d_%.1f_%.1f.gif'%(self.M_x,self.sigma,end_time-start_time), duration=1)

class FDM2d():
    def __init__(self, init_fuc:callable, bcfuc:callable, method='FTCS', sigma_x=0.1, M_x=300, M_y=300, gama=1):
        self.method = method
        self.sigma_x = sigma_x
        self.M_x = M_x
        self.M_y = M_y
        self.delta_x = 1/(M_x-1)
        self.delta_y = 1/(M_y-1)
        self.delta_t = sigma_x*(self.delta_x)**2/gama
        self.sigma_y = gama*self.delta_t/(self.delta_y)**2
        self.all_data = np.zeros((M_x,M_y))
        self.now_step = 0
        self.init_fuc = init_fuc
        self.all_data = init_fuc(self.all_data)
        self.bcfuc = bcfuc
    
    def forward(self,n_step):
        if self.method == 'FTCS':
            for i in range(n_step):
                self.now_step += 1
                temp_data = np.zeros_like(self.all_data)
                temp_data = self.bcfuc(temp_data,self.now_step*self.delta_t)
                for j in range(self.M_x):
                    for k in range(self.M_y):
                        if j==0 or j==self.M_x-1 or k == 0 or k==self.M_y-1: continue
                        temp_data[j,k] = self.sigma_x*self.all_data[j+1,k] + self.sigma_y*self.all_data[j,k+1] +(1-2*self.sigma_x-2*self.sigma_y)*self.all_data[j,k] + self.sigma_x*self.all_data[j-1,k] + self.sigma_y*self.all_data[j,k-1]
                self.all_data = temp_data
    
    def postprocess(self,step_number):
        plt.figure(figsize=(18,8))
        for plotindex in tqdm(range(6)):
            self.forward(step_number[plotindex])
            plt.subplot(2,3,plotindex+1)
            X_plot,Y_plot = np.meshgrid(np.linspace(0,1,self.M_x),np.linspace(0,1,self.M_y))
            plt.contourf(X_plot,Y_plot,self.all_data)
            plt.xlim((0,1))
            plt.ylim((0,1))
            plt.text(0,0,self.method+',boundary condition %d, time step %f\n node number %d, plot at step %d'%(1,self.delta_t,self.M_x,self.now_step))
            plt.xlabel('x')
            plt.ylabel('y')
        plt.savefig('.\\save\\'+self.method+'2d_%d_%f.png'%(self.M_x,self.sigma))

    def gif_create(self, step_interval):
        # reference: https://blog.csdn.net/qingfengxd1/article/details/113725721
        gif.options.matplotlib["dpi"] = 500
        @gif.frame
        def plot():
            #plt.clf()
            X_plot,Y_plot = np.meshgrid(np.linspace(0,1,self.M_x),np.linspace(0,1,self.M_y))
            plt.contourf(X_plot,Y_plot,self.all_data,levels=[0,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
            plt.colorbar()
            plt.text(0,1.1,self.method+',boundary condition %d, time step %f\n node number %d, plot at step %d'%(2,self.delta_t,self.M_x,self.now_step))
            plt.xlim((0,1))
            plt.ylim((0,1))
            plt.xlabel('x')
            plt.ylabel('y')
        frames = []
        start_time =time.time()
        for i in tqdm(range(30)):
            self.forward(step_interval)
            frame = plot()
            frames.append(frame)
        end_time = time.time()
        gif.save(frames, self.method+'2d_%d_%.1f_%.1f.gif'%(self.M_x,self.sigma_x,end_time-start_time), duration=1)


class lwe_FDM():
    '''
    求解一维线性波动方程,c为CFL数
    初值问题，没有边界条件
    '''
    def __init__(self, init_fuc: callable, method=None, c=0.5, M_x=100, a=1):
        if method is None:
            print('Warning! No method is initialized. Default method is Upwind.')
            method = 'Upwind'
        if method not in ['Upwind','Lax_Wendroff','Warming_Beam']:
            raise TypeError
        self.method = method
        self.c = c
        self.a = a
        self.M_x = M_x
        self.delta_x = 1/(M_x-1)
        self.delta_t = c*self.delta_x/a
        self.all_data = np.zeros(M_x)
        self.now_step = 0
        self.init_fuc = init_fuc
        self.all_data = init_fuc(self.all_data)

    def forward(self, n_step):
        if self.method == 'Upwind':
            for i in range(n_step):
                self.now_step += 1
                temp_data = np.zeros_like(self.all_data)
                for j in range(self.M_x):
                    temp_data[j] = (1-self.c)*self.all_data[j] + self.c*self.all_data[(j-1)%self.M_x]
                self.all_data = temp_data
        elif self.method == 'Lax_Wendroff':
            for i in range(n_step):
                self.now_step += 1
                temp_data = np.zeros_like(self.all_data)
                for j in range(self.M_x):
                    temp_data[j] = self.all_data[j] -self.c/2*(self.all_data[(j+1)%self.M_x]-self.all_data[(j-1)%self.M_x])+self.c**2/2*(self.all_data[(j-1)%self.M_x]-2*self.all_data[j]+self.all_data[(j+1)%self.M_x])
                self.all_data = temp_data
        elif self.method == 'Warming_Beam':
            for i in range(n_step):
                self.now_step += 1
                temp_data = np.zeros_like(self.all_data)
                for j in range(self.M_x):
                    temp_data[j] = self.all_data[j] -self.c/2*(3*self.all_data[j]-4*self.all_data[(j-1)%self.M_x]+self.all_data[(j-2)%self.M_x])+self.c**2/2*(self.all_data[(j-2)%self.M_x]-2*self.all_data[(j-1)%self.M_x]+self.all_data[j])
                self.all_data = temp_data
            
    def reinit(self):
        self.now_step = 0
        self.all_data = self.init_fuc(self.all_data)

    def picture(self,plot_time):
        plt.figure(figsize=(18,8))
        for plotindex in range(3):
            if plotindex > 0:
                self.forward(int((plot_time[plotindex]-plot_time[plotindex-1])/self.delta_t))
            else:
                self.forward(int(plot_time[plotindex]/self.delta_t))
            plt.subplot(1,3,plotindex+1)
            plt.plot(np.arange(self.M_x)/(self.M_x-1)-0.5,self.all_data)
            plt.xlim((-0.5,0.5))
            plt.ylim((-0.5,1.5))
            plt.text(0,0.75,self.method+', plot at time %.1f'%(self.now_step*self.delta_t))
            plt.xlabel('x')
            plt.ylabel('u')
        plt.savefig('.\\save\\'+self.method+'_%.1f.png'%(self.now_step*self.delta_t))

    def gif_create(self, step_interval):
        # reference: https://blog.csdn.net/qingfengxd1/article/details/113725721
        gif.options.matplotlib["dpi"] = 500
        @gif.frame
        def plot():
            #plt.clf()
            plt.plot(np.arange(self.M_x)/(self.M_x-1)-0.5,self.all_data)
            plt.text(0,0.75,self.method+', plot at step %d'%(self.now_step))
            plt.xlim((-0.5,0.5))
            plt.ylim((0,2))
            plt.xlabel('x')
            plt.ylabel('u')
        frames = []
        start_time =time.time()
        for i in tqdm(range(40)):
            self.forward(step_interval)
            frame = plot()
            frames.append(frame)
        end_time = time.time()
        gif.save(frames, self.method+'_%.1f.gif'%(self.now_step*self.delta_t), duration=1)


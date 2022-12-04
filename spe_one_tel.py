#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 16:36:22 2022

@author: zrdz
"""


import argparse
from pkg_resources import resource_filename
import os
import numpy as np
import pandas as pd

from cts_core.camera import Camera
from digicampipe.instrument import geometry
from digicampipe.io.event_stream import event_stream

from ctapipe.visualization import CameraDisplay
# from ctapipe.instrument import CameraGeometry
# from ctapipe.image import hillas_parameters, tailcuts_clean

import matplotlib.pyplot as plt
import datetime

import scipy
from scipy.special import factorial
import astropy.units as u



from iminuit import Minuit
#from iminuit.cost import LeastSquares


class mes_fitter:
    def __init__(self,
                 day          = 26,
                 month        = 3,
                 year         = 2022,
                 data_path    = "/net/",
                 tel          = 1,
                 first_file_n = 1,
                 n_files      = 1,
                 plot_dir     = './',
                 save_dir     = './calib/',
                 max_evt      = 10000,
                 dark_baselines = None):
        
        
        self.tel        = tel
        self.date_str   = '{:04d}{:02d}{:02d}'.format(year,month,day)
        self.save_dir   = save_dir
        self.data_path  = data_path
        self.files_path  = os.path.join(data_path,
                                         'cs{}'.format(tel),
                                         'data',
                                         'raw',
                                         '{:04d}'.format(year),
                                         '{:02d}'.format(month),
                                         '{:02d}'.format(day),
                                         'SST1M{}'.format(tel))
        self.first_file_n = first_file_n
        self.n_files      = n_files
        self.plot_dir     = plot_dir
        self.max_evt      = max_evt
        
        self.dark_baselines = dark_baselines
        
        
        digicam_config_file = resource_filename('digicampipe','tests/resources/camera_config.cfg')
        digicam = Camera(_config_file=digicam_config_file)
        self.geom = geometry.generate_geometry_from_camera(camera=digicam)
        self.geom.pix_area = np.ones(self.geom.n_pixels)*482.05 *u.mm**2 ## ??????
        # self.geom.rotate(-90*u.deg)
        
        self.n_pixels = self.geom.n_pixels
        self.pixels   = np.arange(self.n_pixels)
        
        self.save_str = self.date_str+"_{}_{}_Tel{}_".format(first_file_n,
                                                             first_file_n+n_files-1,
                                                             tel)

        self.file_list = []
        for n in range(first_file_n, first_file_n+n_files):
            filepath = os.path.join(self.files_path,'SST1M{}_{}_{:04d}.fits.fz'.format(tel,
                                                                                       self.date_str,
                                                                                       n))
            if os.path.isfile(filepath):
                self.file_list.append(filepath)
            else:
                print('file {} not found'.format(filepath))
        if len(self.file_list) == 0:
            print("Warning : no files")
            
        self.res = None

    def get_histograms(self):
        
        n_bins_adcmax = 50
        n_bins_adcsum = 176
        
        data_stream = event_stream(
            filelist=self.file_list,
            disable_bar = True,
            max_events=self.max_evt
            )
        bins_adcsum = np.linspace(-50,125,n_bins_adcsum)       
        bins_adcmax = np.linspace(0,50,n_bins_adcmax)

        Qsum_hist      =  np.zeros([self.n_pixels,n_bins_adcsum])
        Qmax_hist      =  np.zeros([self.n_pixels,n_bins_adcmax])


        

        tot_evts = 0
        print("starting...to read data")
        
        for ii,event in enumerate(data_stream):
            # for tel in event.r0.tels_with_data:
                tel = 22
                r0data = event.r0.tel[tel]

    
                if r0data._camera_event_type.value==8:
                    tot_evts +=1
                    
                    if self.dark_baselines is None:
                        Qsum = (r0data.adc_samples.T[-15:] - r0data.digicam_baseline).sum(axis=0)
                        Qmax = (r0data.adc_samples.T       - r0data.digicam_baseline).max(axis=0)
                    else:
                        Qsum = (r0data.adc_samples.T[-15:] - self.dark_baselines ).sum(axis=0)
                        Qmax = (r0data.adc_samples.T       - self.dark_baselines ).max(axis=0)
                        

                    i_to_fill_adcsum = np.searchsorted(bins_adcsum[:-1], Qsum)
                    i_to_fill_adcmax = np.searchsorted(bins_adcmax[:-1], Qmax)
                    
                    for pix in self.pixels:
                        Qsum_hist[pix][i_to_fill_adcsum[pix]] = Qsum_hist[pix][i_to_fill_adcsum[pix]] +1
                        Qmax_hist[pix][i_to_fill_adcmax[pix]] = Qmax_hist[pix][i_to_fill_adcmax[pix]] +1
                    
                    
        centers_adcsum = (bins_adcsum[1:]+bins_adcsum[:-1])/2.
        centers_adcmax = (bins_adcmax[1:]+bins_adcmax[:-1])/2.
        print("{} evts proceeded".format(ii+1))
        print("{} evts in histogram".format(tot_evts))
        self.tot_evts = tot_evts
        self.binwidth = centers_adcsum[1]-centers_adcsum[0]
        self.centers_adcsum = centers_adcsum
        self.Qsum_hist      = Qsum_hist
        self.centers_adcmax = centers_adcmax
        self.Qmax_hist      = Qmax_hist
        
        return(centers_adcsum,Qsum_hist,centers_adcmax,Qmax_hist)
    
    
    def spe_spectrum_function(self,x, p, g, x0, sigma_pe, sigma_el):
        STP = np.sqrt(2 * np.pi)
        def Bn(n, p):
            return (n+1)**n * p**n * np.exp(-(n+1)*p) / factorial(n+1)
        
        def single_gauss(x, xn, sigma_n):
            return np.exp(-1/2 *((x-xn)/sigma_n)**2) / (sigma_n * STP)
        
        def sigma_n(n, sigma_pe, sigma_el):
            return np.sqrt(n * sigma_pe**2 + sigma_el**2)
        
        def spe_spectrum_function_(x, p, g, x0, sigma_pe, sigma_el):
    
            S = np.zeros_like(x)
            for n in range(0, 10):
                xn = x0 + n * g
                _sigma_n = sigma_n(n, sigma_pe, sigma_el)
                
                S += (
                    Bn(n, p) * 
                    single_gauss(x, xn, _sigma_n)
                )
            return  S
        
        return spe_spectrum_function_(x, p, g, x0, sigma_pe, sigma_el) * self.tot_evts* self.binwidth
    
    
    
    ##############################################

    
    def aspe_fit(self,pix):
        X = self.centers_adcsum
        Y = self.Qsum_hist[pix][:-1]
        if Y[X>50].sum() >0:
            g0 = 25
        else:
            Y = Y[X<50]
            X = X[X<50]
            g0=0
            
        # X = X[Y>0]
        # Y = Y[Y>0]
        Yerr  = np.sqrt(Y)
        Yerr[Yerr==0] = 1
        # least_squares = LeastSquares(X, Y, Yerr, self.spe_spectrum_function)
        def likelihood(p       ,
                       g       ,
                       x0      ,
                       sigma_pe,
                       sigma_el,
                       ):
            
            preds = self.spe_spectrum_function(X,p,g,x0,sigma_pe,sigma_el)
            # l =  np.sum( [np.log(scipy.stats.poisson(preds[ii]).pmf(Y[ii])) for ii in range(len(Y)) ] )
            l =  np.sum( [np.log(preds[ii])*Y[ii]-preds[ii]-scipy.special.gammaln(Y[ii]+1) for ii in range(len(Y)) ] )
            return -2*l
        
        m = Minuit(likelihood,
                   p        = .05,      # p
                   g        = g0,      # g
                   x0       = 0.,       # x0
                   sigma_pe = 2.,       # sigma_pe
                   sigma_el = 4.,       # sigma_el
                   limit_p        = (0  , 1 ),    ### for old Version of iminuit
                   limit_g        = (0  , 35),
                   limit_x0       = (-10, 10),
                   limit_sigma_pe = (0  , 10),
                   limit_sigma_el = (0  , 20),
                   errordef=0.5,   ### 0.5 for likelihood, 1 for LS
                   # throw_nan=True
                   )
        
        ## new version of Minuit :
        # m.limits['p']  = (0,1)
        # m.limits['g']  = (0,50)
        # m.limits['x0'] = (-10,10)
        # m.limits['sigma_pe'] = (0,20)
        # m.limits['sigma_el'] = (0,20)
        
        m.migrad()
        m.hesse()

        
        param_names = [ 'B_param', 'gain', 'x0', 'sigma_pe', 'sigma_el']
        
        # if m.accurate:
        #     result = dict(zip(param_names, m.np_values() ))
        #     result['P_chi2'] = (((Y - self.spe_spectrum_function(X, *m.np_values())) / Yerr )**2).sum() / (len(X) - len(param_names))
        # else:
        #     print('fit_failed')
        #     result = dict(zip(param_names, [-1,-1,-1,-1,-1]))
        #     result['P_chi2'] = 100
        
        # X = X[Y>0]
        # Y = Y[Y>0]
        # Yerr  = np.sqrt(Y)
        result = dict(zip(param_names, m.np_values() ))
        result['P_chi2'] = (((Y - self.spe_spectrum_function(X, *m.np_values())) / Yerr )**2).sum() / (len(X) - len(param_names))
        
        if result['gain']>12 and result['P_chi2']<10:
            result['calib_flag'] = 1
        else:
            result['calib_flag'] = 0
        return result
######

    def do_all_fit(self):
        results = dict({ 'B_param'    : [],
                         'gain'       : [],
                         'x0'         : [],
                         'sigma_pe'   : [],
                         'sigma_el'   : [],
                         'P_chi2'     : [],
                         'calib_flag' : []})
        
        centers_adcsum,Qsum_hist,centers_adcmax,Qmax_hist = self.get_histograms()
        for pix in self.pixels:
            try :
                result = self.aspe_fit(pix)
                for key in results.keys():
                    results[key].append(result[key])
            except:
                print("pix : {} --> Fit failed".format(pix))
                for key in results.keys():
                    results[key].append(-1)
                    
        self.res            = pd.DataFrame(results)

        self.results = results
        
        return 
    
    def plot_cam_dist(self,save_plots=False):
        
        for key in self.results.keys():
            try:
                f,ax = plt.subplots()
                disp = CameraDisplay(self.geom,ax=ax)
                disp.add_colorbar()
                image = self.res[key].copy()
                # image[self.res['P_chi2']>10] = self.res[key].min()
                disp.image = image
                ax.set_title(key+" Tel {}".format(self.tel))
                if save_plots:
                    f.savefig(self.plot_dir+'cam_{}_tel{}.png'.format(key,self.data_path[-2]))
            except:
                print("Failed")
                return
        plt.show()
            
    def plot_onepix(self,pix): ## todo
        result = dict(self.res.iloc[pix])
        
        # fitlabel = 'fit : \n gain : {:.2} \n Xt : {:.2}'.format(result['gain'],
        #                                                         result['B_param'])
        fitlabel = 'fit : '
        for key in list(result.keys())[:-2]:
            fitlabel=fitlabel+'{}    ::   {:.3} \n'.format(key,result[key])
        fitlabel = fitlabel+' Calib Flag :: {}'.format(result["calib_flag"])
        # print(result)
        f,ax = plt.subplots(figsize=(12,5))
        # ax.plot(self.centers_adcsum,self.Qsum_hist[pix][:-1],label='$\sum$ ADC - baseline')
        ax.fill_between(self.centers_adcsum,
                        self.Qsum_hist[pix][:-1]+(self.Qsum_hist[pix][:-1])**0.5,
                        self.Qsum_hist[pix][:-1]-(self.Qsum_hist[pix][:-1])**0.5,
                        alpha = .5,
                        color = 'green',
                        label='$\sum$ ADC - baseline')
        
        ax.plot(self.centers_adcsum,self.spe_spectrum_function(self.centers_adcsum,
                                                            *list(result.values())[:5]),
                                                            '--',
                                                            color='black',
                                                            label= fitlabel)
        
        def single_gauss(x,A,x0,p,sigma_el):
            return np.exp(-1/2 *((x-x0)/sigma_el)**2) / (sigma_el * np.sqrt(2 * np.pi) -p) *A
        
        A = self.tot_evts* self.binwidth
        ax.plot(self.centers_adcsum,
                single_gauss(self.centers_adcsum,
                            A,
                            result['x0'],
                            result['B_param'],
                            result['sigma_el']),
                ':',
                label='pedestal')
        
        ax.set_yscale('log')
        ax.set_ylim(1e-1,self.Qsum_hist[pix].max()*2)
        ax.grid()
        ax.legend()
        ax.set_xlabel('$\Sigma$ ADC')
        ax.set_title('SPE spectrum -- Tel {} -- pix {}'.format(self.tel,pix))
        
        return f,ax
        
    def save_h5(self,res_save_name=None):
        if res_save_name is None:
            res_save_name = self.save_str
            
        df_param = pd.DataFrame.from_dict(self.results)
        

            
        df_hists = pd.DataFrame(self.Qsum_hist[:,:-1])
        df_hists.columns = self.centers_adcsum
        
        df_param.to_hdf(os.path.join(self.save_dir,
                                     res_save_name+'fitted_parameters.h5'),
                        "df_param")
        df_hists.to_hdf(os.path.join(self.save_dir,
                                     res_save_name+'histograms.h5'),
                        'df_hists')
        
        return
    
    # def read_h5(self,h5file_basename):
    #     h5filepath = os.join(self.save_dir,h5file_basename)
    #     df_hists = pd.read_hdf(h5filepath+'fitted_parameters.h5')
    #     df_param = pd.read_hdf(h5filepath+'histograms.h5')
        
    #     self.results = df_param.to_dict(orient='list')
    #     # self.centers_adcsum = np.array(df_hists.indexes)
    #     self.Qsum_hist = np.array(df_hists)
        
            
#########################################################################################

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot_dir',   type=str, default = './SPE_plots_mc/' )
    parser.add_argument('--data_path',  type=str, default = '/mnt/' )
    parser.add_argument('--day',        type=int, default = 18)
    parser.add_argument('--month',      type=int, default = 7)
    parser.add_argument('--tel',        type=int, default = 1)
    parser.add_argument('--first_file', type=int, default = 208)
    parser.add_argument('--n_file',     type=int, default = 2)
    args = parser.parse_args()
    


    fitter= mes_fitter( day          = args.day,
                        month        = args.month,
                        year         = 2022,
                        data_path    = args.data_path,
                        tel          = args.tel,
                        first_file_n = args.first_file, 
                        n_files      = args.n_file,
                        max_evt      = 50000,
                        plot_dir     = args.plot_dir)
    saveplot = False
    
    

    r = fitter.do_all_fit()
    fitter.save_h5()
    

    
    
    
    ### plot ses for all pix 
    f, ax = plt.subplots(figsize=(12,6))
    for pix in range(fitter.Qsum_hist.shape[0]):
        ax.plot(fitter.centers_adcsum,fitter.Qsum_hist[pix][:-1],color='black',alpha=.1)
    ax.grid()
    ax.set_xlabel('$\Sigma$ ADC')
    ax.set_yscale('log')
    
    

    ## plot hist of param distribution
    mask = fitter.res['calib_flag']==1
    
    kwargs = dict(histtype='stepfilled', alpha=0.3, ec="k")
    # kwargs = dict(histtype='step',alpha=.8)
    for key in fitter.res.keys()[:-1]:
        f,ax = plt.subplots()
        bins = np.linspace(fitter.res[key].min(),
                           fitter.res[key].max(),
                           100)

        ax.hist(fitter.res[key][mask],
                bins=bins,
                label = 'tel2 -- median = {:.3}'.format(np.median(fitter.res[key])),
                **kwargs)
        
        ax.set_title(key)
        ax.legend(loc='upper left')
        ax.grid()
        if key not in[ 'B_param', 'gain','P_chi2']:
            ax.set_xlabel('ADC count')
        if key =='gain':
            ax.set_xlabel('ADC / p.e.')
        if saveplot:
            f.savefig(fitter.plot_dir+'hist_{}.png'.format(key))
    
    
    ## plot one random pix
    pix =np.random.randint(fitter.n_pixels)
    f,ax = fitter.plot_onepix(pix)
    plt.show()
    
    # p0=[
    #     0.06,      # p
    #     22,        # g
    #     0,         # x0
    #     2,         # sigma_pe
    #     6,         # sigma_el
    # ]

    
    

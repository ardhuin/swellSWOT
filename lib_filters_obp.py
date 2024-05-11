import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt
import scipy.constants as consts

def G_field_oneway(theta):
    # question for Alejandro: why 2.25 ? ... when **4 this gives another exp factor ... 
    return np.exp(-np.abs(theta/(np.radians(.107)/2))**2.25*np.log(2))


def get_obp_filter(L_filt = 1, sampling_in = 0.025, f_axis = None, plot_flag = True, kernel="parzen"):
    """
    Get the kernel shape and the spectral response of the filter type specified in the arguments
    
    Arguments:
    L_filt,      Kernel length in km
    sampling_in, Spatial spacing (in km) of the filter kernel
    f_axis,      Frequency axis where to compute the spectral response of the filter
    plot_flag,   Plot filter kernel and spectral response
    kernel,      Kernel to compute ('parzen' or 'bharris' -for Blackman-Harris)
    """
    Nparzen = int(np.round(L_filt/sampling_in))
    if Nparzen%2 == 0:
        Nparzen += 1    
    print("Nb of points OBP kernel: %d" % Nparzen)
    if kernel == "bharris":
        w_obp = sg.blackmanharris(Nparzen) # parzen of 41 points if input sampling is 25 m (kernel length is ~1 km)
    if kernel == "parzen":
        w_obp = sg.parzen(Nparzen) # parzen of 41 points if input sampling is 25 m (kernel length is ~1 km)
    if kernel == "sinc2":
        xn=(np.arange(0,Nparzen-1,1)-Nparzen//2)*2*np.pi
        w_obp = (np.sinc(xn*L_filt*0.00222))**2
    if kernel == "alejandro_azptr":
       PRF = 4332
       N_burst = 9
       v_sat = 7310.
       H = 875.0e3
       f_s = 300.0e6
       v_ground = v_sat/(1+H/6377.0e3)
       lambda_c = (consts.c/35.75e9)
       incidence_deg = 3.5 # just used for the ground/range conversion for xtrack
       d = N_burst*v_ground/PRF # along track posting before OBP averaging
       w_ptr = np.zeros(512)
       w_ptr[:401] = np.sinc(2*v_sat*N_burst/(lambda_c*H*PRF)*np.arange(-200,201)*d)**2
       w_ptr /= np.sum(w_ptr)*d
       w_obp = np.zeros(512)
       y_tab = np.arange(-200,201)*d+1.0e-3
       ant = G_field_oneway(y_tab/H)**4
       w_obp[:len(y_tab)] = np.sin(2*np.pi*v_sat*N_burst/(lambda_c*H*PRF)*y_tab)**2/np.sin(2*np.pi*v_sat/(lambda_c*H*PRF)*y_tab)**2*ant
       w_obp /= np.sum(w_obp)*d
       sampling_in=d/1000.


    w_obp /= np.sum(w_obp)
    x_axis = np.arange(-(len(w_obp)-1)/2, (len(w_obp)-1)/2 + 1)*sampling_in

    # OBP spectrum
    
    if type(f_axis) != np.ndarray:
        if f_axis == None:
            f_obp, S_obp = sg.freqz(w_obp, fs=1/sampling_in)
        else:
            f_obp, S_obp = sg.freqz(w_obp, fs=1/sampling_in, worN=f_axis)
    else:
        f_obp, S_obp = sg.freqz(w_obp, fs=1/sampling_in, worN=f_axis)
    S_obp = np.abs(S_obp)**2

    if plot_flag:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(x_axis, w_obp/np.max(w_obp), ".-")
        ax[1].plot(f_obp, 10*np.log10(S_obp))
        ax[0].grid()
        ax[0].set_xlabel("[km]")
        ax[0].set_title("%s window (not normalized)" % kernel)
        ax[1].grid()
        ax[1].set_xlabel("freq [1/km]")
        ax[1].set_ylabel("dB [m**2.km]")
        plt.show()
    
    return x_axis, w_obp, f_obp, S_obp



def compute_aliased_spectrum_2D(fx_in, fy_in, S_in, fsx, fsy, nrep=2):
    """
    Given the main replica of a spectrum (S_in), compute its aliased version
    for spatial sampling frequencies (fsx, fsy)
    
    Arguments:
    fx_in,   horizontal frequency axis of the main spectrum replica
    fy_in,   vertical frequency axis of the main specumtr replica
    S_in,    main spectrum replica
    fsx,     target sampling frequency of the horizontal axis
    fsy,     target sampling frequency of the vertical axis
    nrep,    number of alias to compute (alias replicas at +-fs, +-2*fs... +-nrep*fs)
    in1side, the spectrum provided 
    
   
    """
    fx_2side = fx_in
    fy_2side = fy_in
    S_2side = S_in
        
    S = S_2side.copy()
    
    Ly, Lx = np.shape(S_2side)    
    fs_idx = np.array([ np.argmin(np.abs(fy_2side - fsy)),\
              np.argmin(np.abs(fx_2side - fsx)) ])
    central_idx = np.array([ np.argmin(np.abs(fy_2side - 0)), \
                   np.argmin(np.abs(fx_2side - 0)) ])
    shift_idx =  fs_idx - central_idx
    # fig, ax = plt.subplots()
    # ax.plot(f_2side, 10*np.log10(S_2side), label='fundamental')
    
    NX, NY = np.meshgrid( np.arange(-nrep, nrep+1), np.arange(-nrep, nrep+1))
    
    for nx, ny in zip(NX.flatten(), NY.flatten()):
        if (nx == ny == 0):
            continue
        shift_idx_nx = np.abs(nx)*shift_idx[1]
        shift_idx_ny = np.abs(ny)*shift_idx[0]
        if (shift_idx_nx >= Lx) or (shift_idx_ny >= Ly):
            continue
        
        #print("Adding replica %d, %d ..." % (nx, ny))
        
        if nx >= 0:
            col_indices_replica = slice(shift_idx_nx, Lx)
            col_indices_in = slice(0, Lx-shift_idx_nx)
        else: # inverse indices   
            col_indices_replica = slice(0, Lx-shift_idx_nx)
            col_indices_in = slice(shift_idx_nx, Lx)
            
        if ny >= 0:
            row_indices_replica = slice(shift_idx_ny, Ly)
            row_indices_in = slice(0, Ly-shift_idx_ny)
        else: # inverse indices   
            row_indices_replica = slice(0, Ly-shift_idx_ny)
            row_indices_in = slice(shift_idx_ny, Ly)

        # print( row_indices_replica[0], row_indices_replica[-1], col_indices_replica[0], col_indices_replica[-1])
        # print( row_indices_in[0], row_indices_in[-1], col_indices_in[0], col_indices_in[-1])
        S[row_indices_replica][:, col_indices_replica] += S_2side[row_indices_in][:, col_indices_in]
    
    return fx_2side, fy_2side, S

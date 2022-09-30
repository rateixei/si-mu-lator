from cycler import cycler

def apply_style(plt):
    plt.rc('text', usetex=False)
    plt.rc('font', family='serif', size=11)
    plt.rcParams["axes.prop_cycle"] = cycler('color', 
        ['#E53D36', '#04628a', '#FFA644', '#998A2F', '#67799E', '#3D372F', '#DBBB00', '#7f7f7f', '#bcbd22', '#17becf'])
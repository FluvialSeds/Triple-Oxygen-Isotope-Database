### TREATISE ON GEOCHEMSITRY DATA ANALYSIS CODE
### JORDON D. HEMINGWAY
### 11. APRIL 2023

#import packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import linregress

#set directory for data
path = '../00 data/'

#define functions
def Dp_d_to_R(Dp17O, d18O, th = 0.5305):
	'''
	Converts Dp17O and d18O values to R values

	Parameters
	----------
	Dp17O : array-like
		Array of Dp17O values, using inputted ref line theta value

	d18O : array-like
		Array of d18O values

	th : float
		Reference line theta; defaults to 0.5305

	Returns
	-------
	R17 : np.array
		Array of corresponding R17 values

	R18 : np.array
		Array of corresponding R18 values
	'''

	#first get R18
	R18 = d18O/1000 + 1

	#then get R17
	R17 = np.exp( Dp17O/1000 + th*np.log(R18) )

	return R17, R18

def R_to_Dp_d(R17, R18, th = 0.5305):
	'''
	Converts R values to Dp17O and d18O values

	Parameters
	----------
	R17 : np.array
		Array of corresponding R17 values

	R18 : np.array
		Array of corresponding R18 values

	th : float
		Reference line theta; defaults to 0.5305

	Returns
	-------
	Dp17O : array-like
		Array of Dp17O values, using inputted ref line theta value

	d18O : array-like
		Array of d18O values
	'''

	#first get d18O
	d18O = 1000*(R18 - 1)

	#then get Dp17O
	Dp17O = 1000*( np.log(R17) - th*np.log(R17) )

	return Dp17O, d18O

def get_line(x,y):
	'''
	'''
	m = (y[1]-y[0])/(x[1]-x[0])
	b = y[0] - x[0]*m

	return m, b

#================#
# THEORY FIGURES #
#================#

# FIG. THEO-1: Self-shielding schematic
#	* following Fig. 6 from Thiemens 2021

# FIG. THEO-2: O3 formation rates as a function of symmetry (Janssen et al. 2001)
#	* following Fig. 9 from Thiemens 2021

#import data
df = pd.read_csv(path+'O3_rxn_rates.csv', encoding='ISO-8859-1')
df = df.dropna()

#make figure
fig,ax = plt.subplots(1,2,
	figsize = (7.48,3),
	sharey = True,
	)

ax[1].set_box_aspect(1)

#make color dict
cs = plt.get_cmap(name = 'Accent', lut = 6)

cd = {'s' : cs.colors[2],
	  'as' : cs.colors[3]
	  }

#~~~~~~~~~~~~~~~~~~~~~~~~#
# PANEL A: RATE BAR PLOT #
#~~~~~~~~~~~~~~~~~~~~~~~~#

#group data by mass
gr = df.groupby('mass')

#calculate number of masses and isotopomers
ni = gr['channel'].count().max()
nm = len(gr)

#make x array and plot
#X = np.arange(nm)


for n,g in gr:

	#make x array
	x = n + np.arange(len(g))*1/ni

	#make color array
	c = [cd[r['sas']] for i,r in g.iterrows()]

	#plot bar plot
	ax[0].bar(
		x, 
		g['k_mean']-1, 
		yerr = g['k_std'],
		bottom = 1,
		color = c,
		edgecolor = 'k',
		width = 1/ni
		)

	#also plot as scatterplot
	ax[0].scatter(
		x, 
		g['k_mean'],
		facecolor = c, 
		edgecolor = 'k',
		linewidth = 0.5,
		s = 50
		)

#add zero line
ax[0].plot(
	[47, 55], 
	[1,1], 
	linewidth = 2, 
	color = 'k', 
	zorder = 0
	)

#set labels and limits
# ax[0].set_ylim([-0.1, 0.55])
ax[0].set_xlim([47.5, 54.5])

ax[0].set_xlabel('mass (amu)')
ax[0].set_ylabel(r'relative formation rate, $k^x/k^{666}$')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# PANEL B: DZPE AND ETA EFFECT #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#calculate regression line
rdf = df[df['in_reg'] == True]
res = linregress(rdf['DZPE'],rdf['k_mean'])

#get symmetric and asymmetric colors
c = [cd[r['sas']] for i,r in df.iterrows()]

# plot data

#first symmetric data
dat = df[df['sas'] == 's']
ax[1].errorbar(
	dat['DZPE'],
	dat['k_mean'],
	yerr = dat['k_std'],
	fmt = 'o',
	mfc = cd['s'],
	mec = 'k',
	ecolor = 'k',
	markersize = 8,
	)


#next, asymmetric data
dat = df[df['sas'] == 'as']
ax[1].errorbar(
	dat['DZPE'],
	dat['k_mean'],
	yerr = dat['k_std'],
	fmt = 'o',
	mfc = cd['as'],
	mec = 'k',
	ecolor = 'k',
	markersize = 8,
	)

#plot regression line
x = np.linspace(-30,30,10)
y = x*res.slope + res.intercept

ax[1].plot(x, y, linewidth = 2, color = 'k')

#set limits and labels
ax[1].set_xlim([-25,25])
ax[1].set_ylim([0.75,1.55])

ax[1].set_xlabel(r'$\Delta(ZPE)$ (cm$^{-1}$)')
# ax[1].set_ylabel(r'relative formation rate, $k^x/k^{666}$')

plt.tight_layout()

#save figure
fig.savefig('Fig_TH_1.pdf',
	bbox_inches = 0,
	transparent = True,
	)

# FIG. THEO-3: Potential energy curve schematic (Heays et al. 2017)
#	* following Fig. 14 from Thiemens 2021??


#===============#
# O-MIF FIGURES #
#===============#

# FIG. O-MIF1: Experimental three-isotope plot
#	* O3 production
#	* O3 dissociation
#	* CO photolysis
#	* CO2 photolysis
#	* H2O2 formation

#import data
df = pd.read_csv(path+'exp_compilation.csv', encoding='ISO-8859-1')
df['dp18O'] = 1000*np.log(df['d18O_mean']/1000 + 1)
df['dp17O'] = 1000*np.log(df['d17O_mean']/1000 + 1)

#make figure
fig,ax = plt.subplots(2,3,
	figsize = (7.48,6)
	)

#flatten it for iterating
ax = ax.flatten()

#make all square
for i in range(len(ax)):
	ax[i].set_box_aspect(1)
	
ax[4].set_xlabel(r"$\delta ' ^{18} O$ (‰ vs. starting)")
ax[0].set_ylabel(r"$\delta ' ^{17} O$ (‰ vs. starting)")

#make color scheme
cs = plt.get_cmap(name = 'Accent', lut = 6)

cm = {
	'electrical': cs.colors[0],
	'microwave': cs.colors[1],
	'photo': cs.colors[4],
	'thermal': cs.colors[3],
	'recombination': cs.colors[5],
	'water_electrolysis': cs.colors[2]
}

#~~~~~~~~~~~~~~~~~~~~~~~~#
# PANEL A: O3 PRODUCTION #
#~~~~~~~~~~~~~~~~~~~~~~~~#

#plot O3 production
ozprod = df[df['experiment_type'].str.contains('ozone_generation')]
ets = set(ozprod['experiment_type'])

for i, et in enumerate(ets):

	#pull color
	c = [val for key, val in cm.items() if key in et][0]

	#get experiments of that type
	temp = ozprod[ozprod['experiment_type'] == et]

	#now separate into O2 and O3
	o2 = temp[temp['compound'] == 'O2']
	o3 = temp[temp['compound'] == 'O3']

	#plot O3
	ax[0].scatter(o3['dp18O'],o3['dp17O'],
		facecolor = c,
		edgecolors = 'k',
		linewidths = 0.5,
		s = 50,
		marker = 'o',
		label = et+'_O3',
		)

	#plot O2
	ax[0].scatter(o2['dp18O'],o2['dp17O'],
		facecolor = 'w',
		edgecolors = c,
		linewidths = 1,
		s = 50,
		marker = 'o',
		label = et+'_O2',
		)

#add MIF and MDF lines
lx = np.array([-100,200])

ax[0].plot(lx, lx,
	linewidth = 2,
	color = 'k',
	label = 'MIF (th = 1)',
	zorder = 0
	)

ax[0].plot(lx, 0.5305*lx,
	'k:',
	linewidth = 2,
	label = 'MDF (th = 0.5305)',
	zorder = 0,
	)

ax[0].set_xlim([-85,150])
ax[0].set_ylim([-85,150])

ax[0].set_title(r'$O_3$ production')

#~~~~~~~~~~~~~~~~~~~~~~~~~~#
# PANEL B: O3 DISSOCIATION #
#~~~~~~~~~~~~~~~~~~~~~~~~~~#

#plot O3 production
ozdiss = df[df['experiment_type'].str.contains('ozone_decomposition')]
ets = set(ozdiss['experiment_type'])

for i, et in enumerate(ets):

	#pull color
	c = [val for key, val in cm.items() if key in et][0]

	#get experiments of that type
	temp = ozdiss[ozdiss['experiment_type'] == et]

	#now separate into O2 and O3
	o2 = temp[temp['compound'] == 'O2']
	o3 = temp[temp['compound'] == 'O3']

	#plot O3
	ax[1].scatter(o3['dp18O'],o3['dp17O'],
		facecolor = c,
		edgecolors = 'k',
		linewidths = 0.5,
		s = 50,
		marker = 'o',
		label = et+'_O3',
		)

	#plot O2
	ax[1].scatter(o2['dp18O'],o2['dp17O'],
		facecolor = 'w',
		edgecolors = c,
		linewidths = 1,
		s = 50,
		marker = 'o',
		label = et+'_O2',
		)

#add MIF and MDF lines
lx = np.array([-100,200])

ax[1].plot(lx, lx,
	linewidth = 2,
	color = 'k',
	label = 'MIF (th = 1)',
	zorder = 0
	)

ax[1].plot(lx, 0.5305*lx,
	'k:',
	linewidth = 2,
	label = 'MDF (th = 0.5305)',
	zorder = 0,
	)

ax[1].set_xlim([-45,95])
ax[1].set_ylim([-45,95])

ax[1].set_title(r'$O_3$ dissociation')


#~~~~~~~~~~~~~~~~~~~~~~~~~#
# PANEL C: H2O2 FORMATION #
#~~~~~~~~~~~~~~~~~~~~~~~~~#

#plot O3 production
perform = df[df['experiment_type'].str.contains('peroxide_formation')]
ets = set(perform['experiment_type'])

for i, et in enumerate(ets):

	#pull color
	c = [val for key, val in cm.items() if key in et][0]

	#get experiments of that type
	temp = perform[perform['experiment_type'] == et]

	#now separate into O2 and H2O2
	o2 = temp[temp['compound'] == 'O2']
	h2o2 = temp[temp['compound'] == 'H2O2']

	#plot H2O2
	ax[2].scatter(h2o2['dp18O'],h2o2['dp17O'],
		facecolor = c,
		edgecolors = 'k',
		linewidths = 0.5,
		s = 50,
		marker = 'o',
		label = et+'_O3',
		)

	#plot O2
	ax[2].scatter(o2['dp18O'],o2['dp17O'],
		facecolor = 'w',
		edgecolors = c,
		linewidths = 1,
		s = 50,
		marker = 'o',
		label = et+'_O2',
		)

#add MIF and MDF lines
lx = np.array([-100,200])

ax[2].plot(lx, lx,
	linewidth = 2,
	color = 'k',
	label = 'MIF (th = 1)',
	zorder = 0
	)

ax[2].plot(lx, 0.5305*lx,
	'k:',
	linewidth = 2,
	label = 'MDF (th = 0.5305)',
	zorder = 0,
	)

ax[2].set_xlim([-30,62])
ax[2].set_ylim([-30,62])

ax[2].set_title(r'$H_2O_2$ production')


#~~~~~~~~~~~~~~~~~~~~~~~~~~#
# PANEL D: CO DISSOCIATION #
#~~~~~~~~~~~~~~~~~~~~~~~~~~#

#plot O3 production
codiss = df[df['experiment_type'].str.contains('CO_decomposition')]
ets = set(codiss['experiment_type'])

for i, et in enumerate(ets):

	#pull color
	c = [val for key, val in cm.items() if key in et][0]

	#get experiments of that type
	temp = codiss[codiss['experiment_type'] == et]

	#now separate into O and CO2
	o = temp[temp['compound'] == 'O']
	co2 = temp[temp['compound'] == 'CO2']

	#plot CO2
	ax[3].scatter(co2['dp18O'],co2['dp17O'],
		facecolor = c,
		edgecolors = 'k',
		linewidths = 0.5,
		s = 50,
		marker = 'o',
		label = et+'_O3',
		)

	#plot O
	ax[3].scatter(o['dp18O'],o['dp17O'],
		facecolor = 'w',
		edgecolors = c,
		linewidths = 1,
		s = 50,
		marker = 'o',
		label = et+'_O2',
		)

#add MIF and MDF lines
lx = np.array([-200,5000])

ax[3].plot(lx, lx,
	linewidth = 2,
	color = 'k',
	label = 'MIF (th = 1)',
	zorder = 0
	)

ax[3].plot(lx, 0.5305*lx,
	'k:',
	linewidth = 2,
	label = 'MDF (th = 0.5305)',
	zorder = 0,
	)

ax[3].set_xlim([-200,5000])
ax[3].set_ylim([-200,5000])

ax[3].set_title(r'$CO$ dissociation')


#~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# PANEL E: CO2 DISSOCIATION #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#plot O3 production
co2diss = df[df['experiment_type'].str.contains('CO2_decomposition')]
ets = set(co2diss['experiment_type'])

for i, et in enumerate(ets):

	#pull color
	c = [val for key, val in cm.items() if key in et][0]

	#get experiments of that type
	temp = co2diss[co2diss['experiment_type'] == et]

	#now separate into O2
	o2 = temp[temp['compound'] == 'O2']

	#plot O2
	ax[4].scatter(o2['dp18O'],o2['dp17O'],
		facecolor = 'w',
		edgecolors = c,
		linewidths = 1,
		s = 50,
		marker = 'o',
		label = et+'_O2',
		)

#add MIF and MDF lines
lx = np.array([-100,100])

ax[4].plot(lx, lx,
	linewidth = 2,
	color = 'k',
	label = 'MIF (th = 1)',
	zorder = 0
	)

ax[4].plot(lx, 0.5305*lx,
	'k:',
	linewidth = 2,
	label = 'MDF (th = 0.5305)',
	zorder = 0,
	)

ax[4].set_xlim([-90,45])
ax[4].set_ylim([-35,100])

ax[4].set_title(r'$CO_2$ dissociation')

#~~~~~~~~~~~~~~~~~~~~~~~~#
# PANEL F: CO2 FORMATION #
#~~~~~~~~~~~~~~~~~~~~~~~~#

#plot O3 production
co2form = df[df['experiment_type'].str.contains('CO2_formation')]
ets = set(co2form['experiment_type'])

for i, et in enumerate(ets):

	#pull color
	c = [val for key, val in cm.items() if key in et][0]

	#get experiments of that type
	temp = co2form[co2form['experiment_type'] == et]

	#now separate into O2
	co2 = temp[temp['compound'] == 'CO2']

	#plot CO2
	ax[5].scatter(co2['dp18O'],co2['dp17O'],
		facecolor = c,
		edgecolors = 'k',
		linewidths = 0.5,
		s = 50,
		marker = 'o',
		label = et+'_O3',
		)

#add MIF and MDF lines
lx = np.array([-100,100])

ax[5].plot(lx, lx,
	linewidth = 2,
	color = 'k',
	label = 'MIF (th = 1)',
	zorder = 0
	)

ax[5].plot(lx, 0.5305*lx,
	'k:',
	linewidth = 2,
	label = 'MDF (th = 0.5305)',
	zorder = 0,
	)

ax[5].set_xlim([10,100])
ax[5].set_ylim([10,100])

ax[5].set_title(r'$CO_2$ formation')

plt.tight_layout()

#save figure
fig.savefig('Fig_O-MIF_1.pdf',
	bbox_inches = 0,
	transparent = True,
	)



# FIG. O-MIF2: Box-and-whisker plots of different slopes
#	A. slopes for all experiments grouped by type
#	B. slopes for CO dissociation grouped by wavelength (for self shielding disc.)
#	C. slopes for O3 dissociation grouped by wavelength

#make figure
fig,ax = plt.subplots(1,3,
	figsize = (7.48,6), #make tall for labels
	sharey = True
	)

#make panels square
for i in range(len(ax)):
	ax[i].set_box_aspect(1)

#~~~~~~~~~~~~~~~~~~~~~~~~~~#
# PANEL A: ALL EXP BY TYPE #
#~~~~~~~~~~~~~~~~~~~~~~~~~~#

#group everything by experiment
g = df.groupby('exp_nr')

#calculate slopes, n, R2, wavelength and type for each experiment
ms = g.apply(lambda v: linregress(v.dp18O,v.dp17O)[0])
ets = g.apply(lambda v: list(set(v['experiment_type']))[0])
r2 = g.apply(lambda v: linregress(v.dp18O, v.dp17O)[2]**2)
lams = g.apply(lambda v: list(set(v['wavelength']))[0])
n = g['dp18O'].count()

#now concatenate these and regroup by experiment type
x = pd.concat([ets, ms, r2, n, lams],axis=1)
x.columns = ['ets', 'ms', 'r2', 'n', 'lam']

#drop experiments with fewer than 3 points or r2 < 0.8
# THIS IS THE FINAL DATASET OF SLOPES TO WORK WITH
scr = x[(x['n'] > 2) & (x['r2'] >= 0.8)]

#groupby experiment time and plot box plots
gr = scr[['ets','ms']].groupby('ets')
gr.boxplot(
	subplots = False,
	rot = 90,
	grid = False,
	ax = ax[0]
	)

ax[0].set_title(r'all experiments by type')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# PANEL B: CO DISS BY WAVELENGTH #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#extract co dissociation experiments
cods = scr[scr['ets'] == 'CO_decomposition_photo']

#then groupby wavelength and plot boxplots
gr = cods[['ms','lam']].groupby('lam')
gr.boxplot(
	subplots = False,
	rot = 90,
	grid = False,
	ax = ax[1]
	)

ax[1].set_title(r'$CO$ photo dissociation')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# PANEL C: O3 DISS BY WAVELENGTH #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#extract co dissociation experiments
ozds = scr[scr['ets'] == 'ozone_decomposition_photo']

#then groupby wavelength and plot boxplots
gr = ozds[['ms','lam']].groupby('lam')
gr.boxplot(
	subplots = False,
	rot = 90,
	grid = False,
	ax = ax[2]
	)

plt.tight_layout()

ax[2].set_title(r'$O3$ photo dissociation')

ax[0].set_ylim([0.5,1.4])

#save figure
fig.savefig('Fig_O-MIF_2.pdf',
	bbox_inches = 0,
	transparent = True,
	)


# FIG. O-MIF3: Three-isotope plot of atmospheric species
#	* Ranges a la Thiemens 2014 Fig. 8
#	A. d'18O vs. d'17O, color coded with MDF and MIF lines
#	B. d'18O vs. D'17O, color coded

#import data
df = pd.read_csv(path+'atmos_compilation.csv', encoding='ISO-8859-1')
df['dp18O'] = 1000*np.log(df['d18O_mean']/1000 + 1)
df['dp17O'] = 1000*np.log(df['d17O_mean']/1000 + 1)
df['Dp17O_5305'] = df['dp17O'] - 0.5305*df['dp18O']

sps = set(df['species'])

#make figure
fig,ax = plt.subplots(1,2,
	figsize = (7.48,4),
	sharex = True
	)

#make panels square
for i in range(len(ax)):
	ax[i].set_box_aspect(1)

#make color scheme
cs = plt.get_cmap(name = 'Paired', lut = 12)

cm = {
	'ox': 'k',
	'oz': cs.colors[1],
	'CO': cs.colors[0],
	'CO2': cs.colors[2],
	'CO3': cs.colors[3],
	'ClO4': cs.colors[4],
	'H2O2': cs.colors[5],
	'H2O': cs.colors[7],
	'N2O': cs.colors[6],
	'NO3': cs.colors[8],
	'SO4': cs.colors[9],
}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# PANEL A: d17O vs. d18O plot #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#loop through and plot
for i, s in enumerate(sps):

	#make temp data frame
	temp = df[df['species'] == s]

	#pull color
	c = [val for key, val in cm.items() if key in s][0]

	#make trop filled, strat open
	if 'trop' in s:
		mfc = c
		mec = 'k'
		zo = 1

	elif 'strat' in s:
		mfc = 'w'
		mec = c
		zo = 0

	#now plot
	ax[0].errorbar(
		temp['dp18O'],
		temp['dp17O'],
		xerr = temp['d18O_std'],
		yerr = temp['d17O_std'],
		fmt = 'o',
		mfc = mfc,
		mec = mec,
		ecolor = 'k',
		markersize = 8,
		zorder = zo,
		label = s,
		)

#add MIF and MDF lines
lx = np.array([-150,250])

ax[0].plot(lx, lx,
	linewidth = 2,
	color = 'k',
	label = 'MIF (th = 1)',
	zorder = 0
	)

ax[0].plot(lx, 0.5305*lx,
	'k:',
	linewidth = 2,
	label = 'MDF (th = 0.5305)',
	zorder = 0,
	)

#set limits and labels
ax[0].set_xlim([-110,260])
ax[0].set_ylim([-60,165])

ax[0].legend(loc = 'best')

ax[0].set_xlabel(r"$\delta ' ^{18}O$ (‰ VSMOW)")
ax[0].set_ylabel(r"$\delta ' ^{17}O$ (‰ VSMOW)")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# PANEL B: D'17O vs. d18O plot #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#loop through and plot
for i, s in enumerate(sps):

	#make temp data frame
	temp = df[df['species'] == s]

	#pull color
	c = [val for key, val in cm.items() if key in s][0]

	#make trop filled, strat open
	if 'trop' in s:
		mfc = c
		mec = 'k'
		zo = 1

	elif 'strat' in s:
		mfc = 'w'
		mec = c
		zo = 0

	#now plot
	ax[1].errorbar(
		temp['dp18O'],
		temp['Dp17O_5305'],
		# xerr = temp['d18O_std'],
		# yerr = temp['d17O_std'],
		fmt = 'o',
		mfc = mfc,
		mec = mec,
		ecolor = 'k',
		markersize = 8,
		zorder = zo,
		)

#set limits and labels
ax[1].set_xlim([-110,260])
ax[1].set_ylim([-4,52])

ax[1].set_xlabel(r"$\delta ' ^{18}O$ (‰ VSMOW)")
ax[1].set_ylabel(r"$\Delta ' ^{17}O_{\theta = 0.5305}$ (‰ VSMOW)")

plt.tight_layout()

#save figure
fig.savefig('Fig_O-MIF_3.pdf',
	bbox_inches = 0,
	transparent = True,
	)


# FIG. O-MIF4: Model-predicted D17O-O2 vs. O2/CO2 ratios
#	* Cao and Bao (2013)
#	* Young et al. (2014)
#	* Liu et al. (2021)

#(MAKE PANELS B AND C IN ILLUSTRATOR)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# PANEL A: CAO AND BAO PREDICTION #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#make equation function
def D17O(rho, tm):

	#first calc. d18O difference
	dd18O = (64 + 146*rho/1.23)/(1 + rho/1.23)

	#then calc Phi
	Phi = 0.519 * dd18O - 7.1738

	#input some constants
	gam = 0.1321
	th = 0.017
	mt = 1.526e19 / 1.09e16 #modern tau
	tau = mt * tm

	#finally get D17O
	D17O = -Phi*gam*th*tau / (1 + rho + gam*th*tau)

	return D17O

#make figure
fig,ax = plt.subplots(1,1, figsize = (4,4))
ax.set_box_aspect(1)

#get colors
cm = plt.get_cmap(name = 'Accent', lut = 6)
cs = [
	cm.colors[0], 
	cm.colors[1], 
	'k', 
	cm.colors[4], 
	cm.colors[3]
	]

#make rho array
lr = np.linspace(-3,3,1000)
rho = 10**lr

#get list of tau multipliers to loop through
tms = [60,10,1,0.5,0.01]

for i, tm in enumerate(tms):

	#calculate D
	D = D17O(rho, tm)

	#plot
	ax.plot(lr, D, linewidth = 2, color = cs[i])


#tighten up labels and axes
ax.set_xlim([-3,3])
ax.set_ylim([-65,0])

ax.set_xlabel(r'$pO_2/pCO_2$')
ax.set_ylabel(r'$\Delta ^{17}O_{0.52}$ (‰ VSMOW)')

plt.tight_layout()

#save figure
fig.savefig('Fig_O-MIF_4A.pdf',
	bbox_inches = 0,
	transparent = True,
	)

# FIG. O-MIF5: Three-isotope plot of all sulfate species
#	A. d18O vs. d17O, sorted by type
#	B. age vs. Dp17O for geologic samples only

# STEP 1: GET DATA FROM ALL LABS ON SAME SCALE
#	1A. ASSUME WOSTBROCK ET AL. 2020 IS "TRUE". FOR LABS WITH UWG-2 AND AIR,
#		DIRECTLY CORRECT TO WOSTBROCK DATA
#	1B. FOR LABS WITHOUT UWG-2 AND AIR, CORRECT TO JOHNSTON-NEW USING SEAWATER
#		SULFATE OR NBS-128 AND UWG-2

#import standards
stds = pd.read_csv(path+'standards.csv', index_col = 0)

#true values
std_true = stds[stds['lab'] == 'Sh']

#get set of labs
labs = set(stds['lab'])

#make empty dataframe to store data in
cal_df = pd.DataFrame(index = labs, columns = ['m','b'])

#loop through and calculate differences from "true" values, only for labs
# where UWG-2 and air exist
for l in labs:

	#get stds from that lab
	std_l = stds[stds['lab'] == l]

	#if no UWG-2 and air, pass
	if not ('UWG-2' in std_l.index and 'air' in std_l.index):
		pass

	else:
		#if they're both there, extract and calc slope
		DD = std_l['Dp17O_5305_mean'] - std_true['Dp17O_5305_mean']
		y = DD[['UWG-2','air']]
		x = std_l.loc[['UWG-2','air'],'d18O_mean']

		#plt.scatter(x,y)
		cal_df.loc[l,:] = get_line(x,y)

#now, loop through and calculate differences from "true" values for labs with
# NBS-127 or seawater sulfate as a 1-point offset, where the "true" values are
# now Johnston Old, corrected to Wostbrock et al. (2020)
std_true = stds[stds['lab'] == 'JO']

#correct Johnston Old to Wostbrock
std_true_new = std_true['Dp17O_5305_mean'] - \
	cal_df.loc['JO','m']*std_true['d18O_mean'] - cal_df.loc['JO','b']

for l in labs:

	#get stds from that lab
	std_l = stds[stds['lab'] == l]

	#pass if already calculated
	if not cal_df.loc[l,:].isnull().any():
		pass

	elif 'NBS-127' in std_l.index:

		DD = std_l['Dp17O_5305_mean'] - std_true_new
		cal_df.loc[l,:] = [0, DD['NBS-127']]

	elif 'Seawater_SO4' in std_l.index:

		DD = std_l['Dp17O_5305_mean'] - std_true_new
		cal_df.loc[l,:] = [0, DD['Seawater_SO4']]


#import data
df = pd.read_csv(path+'so4_compilation.csv', encoding='ISO-8859-1')

#now project correction slope and intercept onto dataframe
t = cal_df.reset_index()
t.columns = ['lab','m','b']
res = pd.merge(df,t,how='left',on='lab')

#calculate corrected Dp17O values
# FILLING NAN d18O VALUES WITH ZERO FOR A CONSTANT OFFSET!
res['Dp17O_5305_corr_mean'] = res['Dp17O_5305_mean'] - \
							  res['m']*res['d18O_mean'].fillna(0) - res['b']

res['dp18O'] = 1000*np.log(res['d18O_mean']/1000 + 1)

#make figure
fig,ax = plt.subplots(1,2,
	figsize = (7.48,2.75)
	)

#make panel A square
ax[0].set_box_aspect(1)

cm = {
	'geologic': [[0,0,0],[1,1,1]],
	'atmospheric': [[0.25,0.25,0.25],[0,0,0]],
	'modern_aquatic': [[0.5,0.5,0.5],[0,0,0]],
	'modern_terrestrial': [[1,1,1],[0,0,0]],
}


sam_type = {
'atmospheric': [
	'Aerosol',
	'Ash',
	'Black_crust',
	'Dust',
	'Marcasite_oxidized',
	'Mirabilite',
	'Varnish',
	'dust',
	'dust_soot',
	],
'geologic': [
	'Anhydrite',
	'Barite',
	'CAS',
	'Evaporite',
	'Gypsum',
	],
'modern_aquatic': [
	'Gypsum_lake',
	'Sulfate_hotspring',
	'Sulfate_ice',
	'Sulfate_lake',
	'Sulfate_marine',
	'Sulfate_river',
	'Sulfate_snow',
	'sulfate_rain',
	],
'modern_terrestrial': [
	'Gypsum_soil',
	'Soil',
	'Terrestrial Sulfates',
	]
}


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# PANEL A: TRIPLE ISOTOPE PLOT #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

for st, li in sam_type.items():

	#get those samples
	temp = res[res['lithology'].isin(li)]

	#plot scatterplot
	ax[0].scatter(
		temp['dp18O'],
		temp['Dp17O_5305_corr_mean'],
		facecolor = cm[st][0],
		edgecolors = cm[st][1],
		linewidths = 0.5,
		s = 50,
		marker = 'o',
		label = st,
		)

#add MDF shading
gmwl_dp18O = np.array([-50,0])
gmwl_dp17O = 0.52654 * gmwl_dp18O #+ 0.014 #Sharp et al. (2018)
gmwl_Dp17O = gmwl_dp17O - 0.5305*gmwl_dp18O

ax[0].fill_between(
	[gmwl_dp18O[0] + 28, gmwl_dp18O[1] + 33], #x values
	[gmwl_Dp17O[0] - 0.15, gmwl_Dp17O[1] - 0.15], #y1 values
	[gmwl_Dp17O[0] - 0.2, gmwl_Dp17O[1] - 0.2], #y2 alues
	alpha = 0.5,
	color = cs.colors[3],
	# zorder = 0,
	)

#set limits and labels
ax[0].set_ylim([-2,6.2])
ax[0].set_xlim([-25,40])

ax[0].set_xlabel(r"$\delta ' ^{18} O$ (‰ VSMOW)")
ax[0].set_ylabel(r"$\Delta ' ^{17} O$ (‰ VSMOW)")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# PANEL B: EARTH HISTORY PLOT #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#make color scheme
cs = plt.get_cmap(name = 'Accent', lut = 6)

cm = {
	'Anhydrite': cs.colors[0],
	'Barite': cs.colors[1],
	'CAS': cs.colors[2],
	'Evaporite': cs.colors[5],
	'Gypsum': cs.colors[4],
}

gs = res[res['lithology'].isin(sam_type['geologic'])]

for li in set(gs['lithology']):

	#get temp
	temp = gs[gs['lithology'] == li]

	#plot results
	ax[1].scatter(
		temp['age_Ma'],
		temp['Dp17O_5305_corr_mean'],
		facecolor = cm[li],
		edgecolors = 'k',
		linewidths = 0.5,
		s = 50,
		marker = 'o',
		label = li,
		)

#add MDF shading
ax[1].fill_between(
	[-100,3500], #x values
	[gmwl_Dp17O[0] - 0.15, gmwl_Dp17O[0] - 0.15], #y1 values
	[gmwl_Dp17O[1] - 0.2, gmwl_Dp17O[1] - 0.2], #y2 values
	alpha = 0.5,
	color = cs.colors[3],
	zorder = 0,
	)

#set limits and labels
ax[1].set_ylim([-1.8,0.3])
ax[1].set_xlim([-100,3350])

ax[1].set_xlabel('age (Ma)')
ax[1].set_ylabel(r"$\Delta ' ^{17} O$ (‰ VSMOW)")


plt.tight_layout()

#save figure
fig.savefig('Fig_O-MIF_5.pdf',
	bbox_inches = 0,
	transparent = True,
	)

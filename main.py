import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import vuba
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib import gridspec
from math import pi
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mpl_toolkits.axes_grid1
import matplotlib.widgets
from matplotlib import animation

def minmax_scale(vals):
    vals = np.asarray(vals)
    m = np.nanmin(vals)
    M = np.nanmax(vals)
    return (vals - m) / (M - m)

# Parameters --------------------------
# Images 
video = vuba.Video('./B3_30Degrees(Spring2017).avi')

# Phenomic data
data = xr.open_dataset('./B3Rbalthica_30Degrees.HDF5')
# -------------------------------------

# Summary data
timepoints = np.arange(0, 146)

area = np.asarray(data['TimeSpecificSummaryData'][:,0][:len(timepoints)])
movement = np.asarray(data['TimeSpecificSummaryData'][:,7][:len(timepoints)])

# Spectral data
indices = {
    13: '0-0.1',
    14: '0.1-0.3',
    15: '0.3-0.5',
    16: '0.5-0.7',
    17: '0.7-0.9',
    18: '0.9-1.2',
    19: '1.2-1.6',
    20: '1.6-1.8',
    21: '1.8-2.2',
    22: '2.2-3.0',
    23: '3.0-4.0',
    24: '4.0-5.0',
    25: '5.0-'
}

categories = []
spectral_data = np.zeros((data['TimeSpecificSummaryData'].shape[0], len(indices.keys())))
for i,ind in enumerate(indices.keys()):
    spectral_data[:,i] = minmax_scale(data['TimeSpecificSummaryData'][:,ind])
    categories.append(indices[ind] + 'Hz')

freq_output_1x1 = data['FreqOutput_1x1']
power_only = np.asarray(freq_output_1x1[:,1,:].T)

bins = len(categories)
spectral_data = spectral_data[:len(timepoints), :]

images = []
for i in range(0, len(video), 600):
    im = video.read(i, grayscale=False)
    images.append(im)

video.close()
images = images[:len(timepoints)]


# Adapted from playbooks like https://stackoverflow.com/a/44989063
class Player(FuncAnimation):
    def __init__(self, fig, func, frames=None, init_func=None, fargs=None,
                 save_count=None, mini=0, maxi=100, pos=(0.125, 0.92), **kwargs):
        self.i = 0
        self.min=mini
        self.max=maxi
        self.runs = True
        self.forwards = True
        self.fig = fig
        self.func = func
        self.setup(pos)
        FuncAnimation.__init__(self,self.fig, self.update, frames=self.play(), interval=10, 
                                           init_func=init_func, fargs=fargs,
                                           save_count=save_count, **kwargs )
    def play(self):
        while self.runs:
            self.i = self.i+self.forwards-(not self.forwards)
            if self.i > self.min and self.i < self.max:
                yield self.i
            elif self.i == self.max:
                self.i = 0
                yield self.i

    def start(self):
        self.runs=True
        self.event_source.start()

    def stop(self, event=None):
        self.runs = False
        self.event_source.stop()

    def forward(self, event=None):
        self.forwards = True
        self.start()
    def backward(self, event=None):
        self.forwards = False
        self.start()
    def oneforward(self, event=None):
        self.forwards = True
        self.onestep()
    def onebackward(self, event=None):
        self.forwards = False
        self.onestep()

    def onestep(self):
        if self.i > self.min and self.i < self.max:
            self.i = self.i+self.forwards-(not self.forwards)
        elif self.i == self.min and self.forwards:
            self.i+=1
        elif self.i == self.max and not self.forwards:
            self.i-=1
        self.func(self.i)
        self.slider.set_val(self.i)
        self.fig.canvas.draw_idle()

    def setup(self, pos):
        playerax = self.fig.add_axes([pos[0],pos[1], 0.64, 0.04])
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(playerax)
        bax = divider.append_axes("right", size="80%", pad=0.05)
        sax = divider.append_axes("right", size="80%", pad=0.05)
        fax = divider.append_axes("right", size="80%", pad=0.05)
        ofax = divider.append_axes("right", size="100%", pad=0.05)
        # sliderax = divider.append_axes("right", size="500%", pad=0.07)
        self.button_oneback = matplotlib.widgets.Button(playerax, label='$\u29CF$')
        self.button_back = matplotlib.widgets.Button(bax, label='$\u25C0$')
        self.button_stop = matplotlib.widgets.Button(sax, label='$\u25A0$')
        self.button_forward = matplotlib.widgets.Button(fax, label='$\u25B6$')
        self.button_oneforward = matplotlib.widgets.Button(ofax, label='$\u29D0$')
        self.button_oneback.on_clicked(self.onebackward)
        self.button_back.on_clicked(self.backward)
        self.button_stop.on_clicked(self.stop)
        self.button_forward.on_clicked(self.forward)
        self.button_oneforward.on_clicked(self.oneforward)

        # Bottom position for figure slider
        sliderax = fig.add_axes([0.15, 0.05, 0.65, 0.03])
        self.slider = matplotlib.widgets.Slider(sliderax, '', self.min, self.max, valinit=self.i)
        self.slider.on_changed(self.set_pos)

    def set_pos(self,i):
        self.i = int(self.slider.val)
        self.func(self.i)

    def update(self,i):
        self.slider.set_val(i)

# Plots -------------
fig = plt.figure(figsize=(12,8), facecolor='black', dpi=100)
gs00 = gridspec.GridSpec(2, 2, figure=fig)
gs01 = gs00[0, 0].subgridspec(2, 1) # Summary plots
gs02 = gs00[0, 1].subgridspec(1, 2) # Radar and live view
gs03 = gs00[1, :].subgridspec(1, 1) # Live spectral heatmap

ax1 = fig.add_subplot(gs01[0,0], frameon=True)
ax2 = fig.add_subplot(gs01[1,0], frameon=True)

ax4 = fig.add_subplot(gs02[0,0], polar=True, frameon=False)
ax5 = fig.add_subplot(gs02[0,1])

ax6 = fig.add_subplot(gs03[0,0], frameon=False)

# Summary plot
for ax in [ax1,ax2]:
  ax.set_xlim([0, len(timepoints)])

ax1.set_ylim([np.nanmin(area), np.nanmax(area)])
ax2.set_ylim([np.nanmin(movement), np.nanmax(movement)])

ax1.set_ylabel('Area', color='white')
ax2.set_ylabel('Movement', color='white')
ax1.set_facecolor('black')
ax2.set_facecolor('black')

ax1.spines['bottom'].set_color('white')
ax1.spines['left'].set_color('white')
ax1.xaxis.label.set_color('white')
ax1.yaxis.label.set_color('white')
ax1.tick_params(axis='x', colors='white')
ax1.tick_params(axis='y', colors='white')

ax2.spines['bottom'].set_color('white')
ax2.spines['left'].set_color('white')
ax2.xaxis.label.set_color('white')
ax2.yaxis.label.set_color('white')
ax2.tick_params(axis='x', colors='white')
ax2.tick_params(axis='y', colors='white')

ax2.set_xlabel('Time (hr)')
ax1.set_title('Summary Measures', color='white')

line1 = ax1.plot(timepoints, area, 'orange', zorder=0)
line2 = ax2.plot(timepoints, movement, 'orange', zorder=0) 

vline1 = ax1.axvline(timepoints[0], color='r', zorder=5)
vline2 = ax2.axvline(timepoints[0], color='r', zorder=5)

sc1 = ax1.scatter(timepoints[0], area[0], color='g', zorder=10)
sc2 = ax2.scatter(timepoints[0], movement[0], color='g', zorder=10) 

# Radar plots
def get_spect_data(val):
  d = spectral_data[val, :].tolist()
  d += d[:1]
  return d

angles = [n / float(bins) * 2 * pi for n in range(bins)]
angles += angles[:1]

ax4.set_xticks(angles[:-1], categories, color='white', size=8)

ax4.set_rlabel_position(0)
ax4.set_yticks([0.25, 0.5, 0.75], ["0.25","0.5","0.75"], color='white', size=7)
ax4.set_ylim(0,1)

ax4.set_theta_offset(pi / 2)
ax4.set_theta_direction(-1)
ax4.set_title('Power spectra at each hour', color='white')


radar1_line, = ax4.plot(angles, get_spect_data(0), 'C1', linewidth=2, linestyle='solid')
radar1_fill, = ax4.fill(angles, get_spect_data(0), 'C1', alpha=0.25)

fills = [radar1_fill]

# Live images
ax5.set_xticks([])
ax5.set_yticks([])

im1 = ax5.imshow(images[0], cmap='gray')

# Spectral heatmap
X = np.linspace(0, 1, len(timepoints))

for i,p in enumerate(power_only):
    power_only[i,:] = minmax_scale(p)

ax6.imshow(power_only, cmap='inferno', aspect='auto')

vline3 = ax6.axvline(timepoints[0], color='r', zorder=5)

# Set y limit (or first line is cropped because of thickness)
ax6.set_xlim(0, len(timepoints))

# No ticks
ax6.set_xticks([])
ax6.set_yticks([])

# ax6.set_ylim(0, bins-1)
ax6.set_xlabel('Time (hr)', color='white')
ax6.set_ylabel('Frequency', color='white')
ax6.set_title('Power Spectral Landscape', color='white')


# The function to be called anytime a slider's value changes
def update(val):
  vline1.set_xdata([val, val])
  vline2.set_xdata([val, val])

  sc1.set_offsets([timepoints[val], area[val]])
  sc2.set_offsets([timepoints[val], movement[val]])

  radar1_line.set_ydata(get_spect_data(val))

  fills[0].remove()
  fills[0], = ax4.fill(angles, get_spect_data(val), 'C1', alpha=0.25)

  im1.set_array(images[val])

  vline3.set_xdata([val, val])

plt.subplots_adjust(
    top=0.88,
    bottom=0.12,
    left=0.075,
    right=0.95,
    hspace=0.245,
    wspace=0.205
)

ani = Player(fig, update, maxi=len(timepoints)-1, pos=[0.15, 0.95])

plt.show()
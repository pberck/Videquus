# Run grand on data
# Uses the updated grand code (w/o hacks)
#
import matplotlib as mpl
mpl.use('qt5agg') #https://github.com/matplotlib/matplotlib/issues/9637

import numpy as np
import random
import sys
import pandas as pd
from datetime import datetime, timedelta
import os
from shlex import quote

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import argparse
import pprint

import scipy
import operator
import time

#import dtw
from sklearn.metrics.pairwise import manhattan_distances

from grand import IndividualAnomalyTransductive
from grand import IndividualAnomalyInductive

parser = argparse.ArgumentParser()
parser.add_argument( '-b', "--block", action='store_true', default=False,
                     help='Keep showing the plot until closed' )
parser.add_argument( '-c', "--cam_id", type=int, default=11,
                     help='Camera ID' )
#parser.add_argument( '-f', '--files', action='append', help='Data files')
parser.add_argument( '-C', "--col_name", type=str, default="average_movement",
                     help='Column to use from data (average_movement, exp_weighted_moving_average, alarm_average_movement)' )
#
parser.add_argument( '-p', "--p_value", type=float, default=0.6,
                     help='P-value' )
parser.add_argument( "--ref_group", type=str, default="hour-of-day",
                     help='Reference group' )
parser.add_argument( "--dev_threshold", type=float, default=0.6,
                     help='Threshold on dev level' )
parser.add_argument( "--measure", type=str, default="median",
                     help='Strangeness, median, knn or lof.' )
parser.add_argument( "--martingale", type=int, default=15,
                     help='w_martingale parameter for Grand' )
parser.add_argument( "--type", type=str, default="T",
                     help='Transductive T (default) or inductive I' )
#
parser.add_argument( '-r', "--resample_time", type=str, default="60s",
                     help='Resample time' )
parser.add_argument( '-s', "--samples", type=int, default=8888,
                     help='Number of samples' )
parser.add_argument( '-d', "--days", type=int, default=7,
                     help='Days to calculate normal on' )
parser.add_argument( '-n', "--non_conformity", type=str, default="median",
                     help='Non-conformity (median, knn, lof)' )
#
parser.add_argument( "--start_dt", type=str,       default="2018-08-04 22:00",
                     help='Start date of reference' )
parser.add_argument( "--end_dt", type=str,         default="2018-08-08 08:00",
                     help='Start date of reference' )
parser.add_argument( "--train_start_dt", type=str, default="2018-08-01 22:00",
                     help='Start date of training' )
parser.add_argument( "--train_end_dt", type=str,   default="2018-08-03 08:00",
                     help='Start date of training' )
#parser.add_argument( "--start_dt", type=str,       default="2018-08-18 22:00", help='Start date of reference' )
#parser.add_argument( "--end_dt", type=str,         default="2018-08-24 08:00", help='Start date of reference' )
#parser.add_argument( "--train_start_dt", type=str, default="2018-08-10 22:00", help='Start date of training' )
#parser.add_argument( "--train_end_dt", type=str,   default="2018-08-18 08:00", help='Start date of training' )
#
parser.add_argument( "--do_train", action='store_true', default=False,
                     help='Run training step' )
#
parser.add_argument( '-H', "--high_only", action='store_false', default=True,
                     help='Show high deviations only' )
parser.add_argument( '-X', "--extra_text", type=str, default=None,
                     help='Extra text in filename' )
#
parser.add_argument( '-R', "--random_data", action='store_true', default=False,
                     help='Use (pseudo)random data' )

args = parser.parse_args()

# Save what we do for later including plots created
invocation = "python3 " + ' '.join(quote(s) for s in sys.argv)
with open( "invocation_log.txt", "a") as f:
    f.write( invocation + "\n" )

class TSERIES():
    '''
    Base class for the time series
    '''
    def __init__(self):
        self.data = None

class RSERIES(TSERIES):
    '''
    Random time series, for testing only.
    '''
    def __init__(self):
        self.start_dt = pd.Timestamp(2018, 1, 1, 0)
        self.end_dt   = pd.Timestamp(2018, 12, 31, 0)
        self.freq     = pd.Timedelta(seconds=300)
        self.dt_index = pd.date_range( self.start_dt, self.end_dt, freq=self.freq )
        print( "---------------------------------------------------------------------" )
        print( "- RANDOM SERIES                                                     -" )
        print( "---------------------------------------------------------------------" )
    def generate_samples(self, cam=11, rs="300s"):
        '''
        Generate one data sample (eg movement for next frame).
        '''
        #
        args.resample_time
        self.dt_index = pd.date_range( self.start_dt, self.end_dt, freq=rs ) 
        for x in self.dt_index:
            if x.hour not in [22,23,0,1,2,3,4,5]:
                continue
            if x.day == 8 and x.hour == 3: # more movement 3 till 4
                yield x, [0.8 * np.random.random_sample()/4]
            elif x.day == 10 and x.hour == 4: # less
                yield x, [0.1 * np.random.random_sample()/4]
            elif x.date != 8 and x.date != 10 and x.hour == 4: # more movement 4 till 5
                yield x, [0.8 * np.random.random_sample()/4]
            else:
                yield x, [0.2 * np.random.random_sample()/4]

# "id","camera_unit_id","average_movement","exp_weighted_moving_average","timestamp","load1","load5","load15","wifi_strength","cpu_temperature","alarm_moving_average","created","wifi_strength_dbm"
# "5618446","8","0","4.61983e-14","2018-08-01 00:01:00","0.96","1.07","1.14",70,"68.218","0.000320511","2018-08-01 00:01:14.934",-36
class CAMSERIES(TSERIES):
    '''
    The webcam movement data, from the CameraMinuteSensorData.csv file.
    Provides a generator to yield nightly data (22:00 - 06:00)
    '''
    def __init__(self, fn):
        self.filename = fn
        self.df = pd.read_csv( fn ) # "CameraMinuteSensorData.csv"
        print( list(self.df.columns) )
        self.df['ts'] = pd.to_datetime(self.df['timestamp'].astype(str), format='%Y-%m-%d %H:%M:%S')
        print( "CAMSERIES", self.df.shape )
        self.camids = self.df["camera_unit_id"].value_counts().index.values
        print( self.camids )
        #
        self.start_dt   = self.df["ts"].min()
        self.end_dt     = self.df["ts"].max() 
        self.start_date = str(self.start_dt)[0:10]
        self.end_date   = str(self.end_dt)[0:10] 
        print( "CAMSERIES", self.start_date, "-", self.end_date )
    def generate_samples(self, cam=0,
                         col=args.col_name,
                         rs=None,
                         hours=[22,23,0,1,2,3,4,5]):  # JUST THE NIGHT TIME
        '''
        Generate one data sample (eg movement for next frame).
        '''
        if cam == 0:
            col_df = self.df
        else:
            col_df = self.df.loc[ self.df["camera_unit_id"] == cam ]
        #
        col_df.sort_values(by=["ts"], inplace=True)
        print( col_df.head() )
        #
        col_ts = pd.Series(data = col_df[col].values, index = col_df['ts'])
        print( col_ts.head() )
        if rs:
            col_ts = col_ts.resample(args.resample_time).sum() # resample into larger time periods
            #col_ts = col_ts.resample(args.resample_time).mean() # resample into larger time periods
            print( col_ts.head() )
        #
        for i,x in col_ts.iteritems():
            if i.hour in hours:
                yield i, [x]
    def generate_slices(self, cam=0, start_slice=0, start_inc=0, slice_size=0, end_slice=0, col=None):
        # Maybe the generated slice should have 'X' and 'Y' as column names...
        '''
        Generate dataframes.
        start_slice: date/time; start the first slice at this date/time point.
                     timedelta; start the first slice at this time on the first day in the data.
                     0; start at the first data point in the data.
        start_inc:   time to add to start_slice to start next slice. For example, "1d" to take daily slices.
                     0 returns consecutive slices.
        slice_size:  time span for the slice. Cannot be 0, this argument must be set.
        end_slice:   date/time, or 0 for the last one in the data set.

        example:
                     start_slice=pd.Timedelta("22:00:00"), # start on first day at 22:00
                     slice_size=pd.Timedelta('4h'),        # take 4 hours of data per slice
                     start_inc=pd.Timedelta('24h'),        # next slice starts 1 day later
                     --> this takes slices between 22:00 and 02:00
        '''
        if cam == 0:
            col_df = self.df
        else:
            col_df = self.df.loc[ self.df["camera_unit_id"] == cam ]
        #print( col_df.head() )
        #
        if type(start_slice) == type(pd.Timedelta("1d")):
            start_dt = self.df["ts"].min()
            start_dt.replace(hour=0, minute=0, second=0)
            start_slice = start_dt + start_slice - pd.Timedelta("1m") # why is this kludge needed?
        if start_slice == 0:
            start_slice = self.df["ts"].min()
        else:
            pass # keep supplied
        #
        if end_slice == 0:
            end_slice = self.df["ts"].max()
        else:
            pass # take the supplied one
        if start_inc == 0:
            start_inc = slice_size #slice_size should not be 0...
        end_curr_slice = start_slice + slice_size
        while end_curr_slice < end_slice:
            print( start_slice, end_curr_slice )
            slice_df       = col_df[ (col_df["ts"] >= start_slice) & (col_df["ts"] < end_curr_slice) ]
            start_slice   += start_inc
            end_curr_slice = start_slice + slice_size
            print( "len(slice_df)", len(slice_df), slice_df.shape )
            yield slice_df

def plot(results_df, seq, fn_str=None, xtra=None):
    '''
    Plot a time series and cosmo output data.
    '''
    strangeness = True
    if strangeness:
        fig0, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(12,6))
        ax0 = axes[0]
        ax1 = axes[1]
        ax2 = axes[2]
    else:
        fig0, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(12,6))
        ax0 = axes[0]
        ax1 = axes[1]
    fig0.suptitle( "cam:"+str(args.cam_id)+" dev_threshold:"+str(args.dev_threshold)+" "+args.ref_group +
                   " p-val:"+str(args.p_value)+" non_conformity:"+args.measure+ " martingale:"+str(args.martingale) +
                   " type:"+args.type )
    ax0.set_xlabel("Time")
    #ax0.set_ylabel("Data")
    ax0.set_ylabel( args.col_name )
    #ax0.plot( results_df["dt"].values, results_df["x"].values )
    colours = np.where(results_df["isdevp"].values > 0.1, 'r', "#1d6fa8") 
    ax0.vlines( x=results_df["dt"].values,
                ymin=0, ymax=results_df["x"].values,
                color=colours, alpha=0.8) #"#1d6fa8")
    #ax0.plot( results_df["dt"].values, results_df["strangeness"].values )
    #
    ax1.plot( results_df["dt"].values, results_df["deviation"].values, ".", alpha=0.5 )#, markersize=6 )
    #ax1.vlines( x=results_df["dt"].values,
    #            ymin=0, ymax=results_df["deviation"].values,
    #            color="#1d6fa8", alpha=0.5)
    ax1.set_ylabel("deviation")
    colourspval = np.where(results_df["pvalue"].values < 0.1, 'g', 'g') # these can be different
    ax1.scatter( results_df["dt"].values, results_df["pvalue"].values,
                 alpha=0.25, marker=".", color=colourspval, label="p-value" ) #, s=4 )
    ax1.scatter( results_df["dt"].values, results_df["isdevp"].values,
                 alpha=0.25, marker=".", color="red", label="isdevp" )
    ax1.axhline(y=args.dev_threshold, color='r', linestyle='--', label="Threshold", alpha=0.5)
    #
    if strangeness:
        ax2.plot( results_df["dt"].values, results_df["strangeness"].values, "." )
        ax2.set_ylabel("strangeness")
        #ax0.plot( results_df["dt"].values, results_df["strangeness"].values, "." )
        ax2.vlines( x=results_df["dt"].values,
                    ymin=0, ymax=results_df["x"].values,
                    color=colours, alpha=0.4) #"#1d6fa8")

    #
    fig0.autofmt_xdate()
    if not fn_str:
        cols = "".join([c[0] for c in args.col_name.split("_")])
        fn_str = "cam"+str(args.cam_id)+"dthresh"+str(args.dev_threshold)+"_"+args.ref_group+"_pval"+str(args.p_value)+"_"+args.measure+"_mgale"+str(args.martingale)+"_t"+args.type+"_C"+cols
    if xtra:
        if strangeness:
            ax2.set_xlabel( xtra )
        else:
            ax1.set_xlabel( xtra )
        fn_str += "_"+xtra.replace(" ", "")
    if seq > -1:
        fn_str += "_seq"+str(seq)
    if args.do_train:
        fn_str += "_TR"
    if args.extra_text:
        fn_str += "_"+args.extra_text
    if os.path.exists( "grand1_"+fn_str+".png" ):
        os.remove( "grand1_"+fn_str+".png" )
    fig0.savefig("grand1_"+fn_str+".png", dpi=300)
    print( "Saved", "grand1_"+fn_str+".png" )
    with open( "invocation_log.txt", "a") as f:
        f.write( "  "+"grand1_"+fn_str+".png\n" )

def plot_plain(results_df, seq, fn_str=None, xtra=None):
    '''
    Plot movement data only, without the cosmo data.
    '''
    fig0, ax0 = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(12,6))
    if seq > -1:
        fig0.suptitle( "cam:"+str(args.cam_id)+", seq "+str(seq) )
    else:
        fig0.suptitle( "cam:"+str(args.cam_id) )
    ax0.set_xlabel("Time")
    #ax0.set_ylabel("Data")
    ax0.set_ylabel( args.col_name )
    #ax0.plot( results_df["dt"].values, results_df["x"].values )
    colours = "#1d6fa8"
    ax0.vlines( x=results_df["dt"].values,
                ymin=0, ymax=results_df["x"].values,
                color=colours, alpha=0.8) #"#1d6fa8")
    #
    fig0.autofmt_xdate()
    if not fn_str:
        cols = "".join([c[0] for c in args.col_name.split("_")])
        fn_str = "cam"+str(args.cam_id)+"_plain"+"_C"+cols
    if args.do_train:
        fn_str += "_TR"
    if xtra:
        fn_str += "_"+xtra.replace(" ", "")
    if seq > -1:
        fn_str += "_seq"+str(seq)
    if args.extra_text:
        fn_str += "_"+args.extra_text
    if os.path.exists( "grand1_"+fn_str+".png" ):
        os.remove( "grand1_"+fn_str+".png" )
    fig0.savefig("grand1_"+fn_str+".png", dpi=300)
    print( "Saved", "grand1_"+fn_str+".png" )
    with open( "invocation_log.txt", "a") as f:
        f.write( "  "+"grand1_"+fn_str+".png\n" )

'''
q = RSERIES()
qgen = q.generate_samples()
for x in range(0,24):
    x, y =  next(qgen)
    print( x, y )
sys.exit(1)
'''
if not args.random_data:
    bb = CAMSERIES( "CameraMinuteSensorData.csv" )
else:
    bb = RSERIES()

'''
generator = bb.generate_slices(cam=11,
                               start_slice=pd.to_datetime("2018-08-01 22:00", format='%Y-%m-%d %H:%M'),
                               #=pd.Timedelta("22:00:00"),
                               slice_size=pd.Timedelta('8h'),
                               start_inc=pd.Timedelta('1d'),
                               end_slice=pd.to_datetime("2018-08-08 08:00", format='%Y-%m-%d %H:%M') )
#print( [x.head(2) for x in generator] )
train_data = [x for x in generator]
'''
#night_data = [x for x in generator]
#print( night_data ) # array with dataframes
#sys.exit(1)

generator = bb.generate_samples( cam=args.cam_id, rs=args.resample_time )

# Choose between the Transductive or Inductive version.
ref_groups_list = args.ref_group.split(",")
if args.type == "T":
    indev = IndividualAnomalyTransductive(w_martingale=args.martingale,  # Window size for computing the deviation level
                                          non_conformity=args.measure,   # Strangeness measure: "median","knn","lof"
                                          k=15,                          # Used if non_conformity is "knn"
                                          dev_threshold=args.dev_threshold, # Threshold on the deviation level
                                          ref_group=ref_groups_list,
                                          #ref_group="external",
    ) # reference group construction: "week", "month", "season", "external"
else:
    indev = IndividualAnomalyInductive( w_martingale=args.martingale,# Window size for computing the deviation level
                                        non_conformity=args.measure, # Strangeness measure: "median" or "knn" or "lof"
                                        k=50,                        # Used if non_conformity is "knn"
                                        dev_threshold=args.dev_threshold)

# Train
# Training consists of pushing one week of data through the algorithm.
# (Pretending we are "live")
# Can be switched off.
#
if args.do_train:
    print( "--------" )
    print( "Training" )
    print( "--------" )
    inside    = False
    processed = 0
    training  = []
    sequence  = 0
    train_start_dt = pd.to_datetime(args.train_start_dt, format='%Y-%m-%d %H:%M')
    train_end_dt   = pd.to_datetime(args.train_end_dt,   format='%Y-%m-%d %H:%M')
    print( args.train_start_dt, "--", args.train_end_dt )
    prev_dt = train_start_dt
    while True: # since we train outside of the loop, this can/should be simplified with a big select...
        dt, x = next(generator)
        if dt < train_start_dt:
            continue
        if not inside:
            print( "New sequence", sequence, str(dt) )
            inside = True
        if dt >= train_end_dt:
            print( "End sequence", sequence, str(dt) )
            break # before hour check, otherwise we always get 1 in start of next day
        curr_dt = dt

        # To know we have started a new night, the jump is from 5:59 to 22:00
        # Take an arbitrary 1 hour gap to start new night
        if pd.Timedelta(dt - prev_dt).seconds > 3600:
            print( "End sequence", sequence, str(prev_dt) )
            sequence += 1
            print( "New sequence", sequence, str(dt) )
        prev_dt = dt

        # "Train"
        if args.type == "T": # for IndividualAnomalyTransductive
            devContext = indev.predict(dt, x) #maybe it is enough to just push the data in?
            st, pv, dev, isdev = devContext.strangeness, devContext.pvalue, devContext.deviation, devContext.is_deviating
            training.append( [ dt, x[0], st, pv, dev, isdev, sequence ] )
        else:
            training.append( [ dt, x[0], 0, 0, 0, 0, sequence ] )
        processed += 1

    print( "processed, dt, x", processed, dt, x )
    #indev.fit( np.array( [[1,2,3], [4,5,6], [7,8,9]] ) )
    training_df = pd.DataFrame( training, columns=["dt", "x", "strangeness", "pvalue", "deviation", "isdevp", "seq"] )
    if args.type == "I":
        indev.fit( training_df["x"].values.reshape(-1, 1) )
    #
    plot( training_df, -1, xtra="Training Data" )
    plot_plain( training_df, -1, xtra="Training Data" )
    plot_plain( training_df[(training_df["seq"]==1)], 1, xtra="Training Data" )

    # Plot a vline plot for each day, each day starting from 0+daynr to compare
    # the same time period over multiple days (they will be next to each other).
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,4))
    lines = []
    #sequence_list = sorted(np.unique(training_df["seq"].values))
    #print( sequence_list )
    cs = [ "#1c641c", "#217821", "#278c27", "#2ca02c", "#32b432", "#38c838", "#4ccd4c", "#60d360" ] #greens
    for seq in range(0, sequence+1):
        plot_df = training_df[(training_df["seq"]==seq)]
        xticks = np.arange(0+seq, (plot_df.shape[0]*(sequence+2))+seq, sequence+2) # +2 to get spacing between groups
        lws = np.where( plot_df["isdevp"].values > 0.1, 1, 1 )
        colours = np.where( plot_df["isdevp"].values > 0.1, "#2ca02c", "#2ca02c") #https://www.colorhexa.com/ff7f0e
        #colours = cs[seq%len(cs)] # not much difference anyway
        line = ax.vlines( xticks, ymin=0, ymax=plot_df["x"].values,
                          color=colours,
                          lw=lws
        )
        lines.append( str(seq) )
    fn_str = "cam"+str(args.cam_id)+"dthresh"+str(args.dev_threshold)+"_"+args.ref_group+"_pval"+str(args.p_value)+"_"+args.measure+"_mgale"+str(args.martingale)+"_t"+args.type+"_overview_train"
    if args.extra_text:
        fn_str += "_"+args.extra_text
    if os.path.exists( "grand1_"+fn_str+".png" ):
        os.remove( "grand1_"+fn_str+".png" )
    fig.savefig("grand1_"+fn_str+".png", dpi=300)
    print( "Saved", "grand1_"+fn_str+".png" )
    with open( "invocation_log.txt", "a") as f:
        f.write( "  "+"grand1_"+fn_str+".png\n" )
    plt.show(block=args.block)
    if args.block:
        plt.pause(1)
    print( "--------------" )
    print( "Training ready" )
    print( "--------------" )

# ------------------------------------------------------------------------
# Here we reset out data, and start feeding the "live" data
# stream to the algorithm. Produce a nightly plot.
# ------------------------------------------------------------------------
#
generator = bb.generate_samples( cam=args.cam_id, rs=args.resample_time )

inside   = False
counter  = args.samples
my_dt    = pd.Timestamp(2018, 1, 1, 0)
my_td    = pd.Timedelta(args.resample_time)
results  = []
start_dt = pd.to_datetime(args.start_dt, format='%Y-%m-%d %H:%M')
end_dt   = pd.to_datetime(args.end_dt,   format='%Y-%m-%d %H:%M')
curr_dt  = start_dt
prev_dt  = start_dt
sequence = 0
print( start_dt, "--", end_dt )
for dt, x in generator:
    if dt < start_dt:
        continue
    if not inside:
        print( "New sequence", sequence, str(dt) )
        inside = True
    if dt >= end_dt:
        print( "End sequence", sequence, str(dt) )
        break # before hour check, otherwise we always get 1 in start of next day
    curr_dt = dt

    # To know we have started a new night, the jump is from 5:59 to 22:00
    if pd.Timedelta(dt - prev_dt).seconds > 3600:
        print( "End sequence", sequence, str(prev_dt) )
        sequence += 1
        print( "New sequence", sequence, str(dt) )
    prev_dt = dt
    
    #devContext = indev.predict(dt, x)
    #dt = my_dt
    devContext = indev.predict(dt, x)
    my_dt += my_td
    st, pv, dev, isdev = devContext.strangeness, devContext.pvalue, devContext.deviation, devContext.is_deviating
    if args.high_only or isdev:
        # unix timestamp to find video sequence. Do I need to add summertime offset, 1 or 2 hrs?
        # tl_20180903T062836Z-1536886119754.ts.mp4
        # -->         tl_XXXZ-1533939960000.ts
        #tz_seconds = -time.timezone
        #epoch = f"tl_XXXZ- {(dt+tz_seconds).strftime('%s')} 000.ts"
        epoch = f"tl_XXXZ- {dt.strftime('%s')} nnn.ts"
        print("Time: {} ==> strangeness: {:.5f}, p-value: {:.5f}, deviation: {:.5f} ({})  {}".format(
            dt, st, pv, dev, "high" if isdev else "low", epoch)
        )
    results.append( [ dt, x[0], st, pv, dev, isdev, sequence ] ) # note the x[0]
    counter -= 1
    if counter < 0:
        break

results_df = pd.DataFrame( results, columns=["dt", "x", "strangeness", "pvalue", "deviation", "isdevp", "seq"] )
print( results_df )
#print( results_df["dt"].values )
#print( results_df["x"].values )

plot( results_df, -1 )

# ------------
# Plot a vlines diagram with data grouped per period (22-06).
# The data marked as anomaly is in orange, the rest is in blue.
# TODO: fix a-axis (show only time?) Fake time axis, use strings?
# ------------
#
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,4))
lines = []
for seq in range(0, sequence+1):
    plot_df = results_df[(results_df["seq"]==seq)]
    #print( plot_df )
    plot( plot_df, seq )
    #line, = ax.plot( plot_df["x"].values )
    #print( plot_df["dt"].values )
    xticks = np.arange(0+seq, (plot_df.shape[0]*(sequence+2))+seq, sequence+2) # +2 to get spacing between groups
    lws = np.where( plot_df["isdevp"].values > 0.1, 1, 1 )
    colours = np.where( plot_df["isdevp"].values > 0.1, "#ff7f0e", "#0e8eff") #https://www.colorhexa.com/ff7f0e
    line = ax.vlines( xticks, ymin=0, ymax=plot_df["x"].values,
                      color=colours,
                      lw=lws
    )
    lines.append( str(seq) )
    #histograms.append( plot_df["x"].values )
#ax.legend(lines) # just shows 0--sequence

cols = "".join([c[0] for c in args.col_name.split("_")])
fn_str = "cam"+str(args.cam_id)+"dthresh"+str(args.dev_threshold)+"_"+args.ref_group+"_pval"+str(args.p_value)+"_"+args.measure+"_mgale"+str(args.martingale)+"_t"+args.type+"_C"+cols+"_overview"
if args.extra_text:
    fn_str += "_"+args.extra_text
if os.path.exists( "grand1_"+fn_str+".png" ):
    os.remove( "grand1_"+fn_str+".png" )
fig.savefig("grand1_"+fn_str+".png", dpi=300)
print( "Saved", "grand1_"+fn_str+".png" )
with open( "invocation_log.txt", "a") as f:
    f.write( "  "+"grand1_"+fn_str+".png\n" )

plt.show(block=args.block)

**Julia’s guide to sleep scoring**

**Major criteria for sleep scoring:**

1) Defining NREM is easy because EEG has much higher amplitude with lower frequency, EMG is low, delta power is very high, some theta power, but T:D ratio is low because they are similar.  
2) Wake: EEG amplitude is low, EMG is high, delta should be basically 0, T:D ratio can be decent.  
3) REM: similar EEG to wakefulness (low amplitude, high frequency). EMG should be basically flat, theoretically lower than in NREM because of muscle atonia. There should be a peak in T:D ratio.

**Sleep sequences:** 

1) REM always comes after NREM and always finishes with wakefulness (not everyone does this, some people allow REM to go straight back to NREM). But be consistent with your policy.  
2) Can have some EMG “flinches” during REM that wouldn’t be considered waking.   
3) Cannot go from wake straight back to REM: wake always goes through NREM.

**To analyse data in Spike 7:**

1) Make new copy of original data file  
2) Name “Sleep Day 1”  
3) Choose best EEG channel and hide all but EMG and EEG.  
4) Treat EEG channel:  
   1) IIR filter, low pass second order, Butterworth, pass below 0.5 Hz, new virtual channel, rename this channel EEGlow  
   2) Create new virtual channel, matched to original EEG channel. Expression: EEGorig \- EEGlow, rename EEGfilt  
5) Treat EMG channel:  
   1) FIR filter, band pass filter, pass between 5 and 45 Hz, transition gap 1.8, new virtual channel, remane EMGfilt.Run OSD script.  
6) Hide all channels except EEGfilt and EMGfilt. These channels should both be centred around 0 on y-axis.  
7) Offline, banded power, EMG RMS time constant \= 5s, EEG smoothing time constant \= 5s. Delta \= 0-4 Hz; Theta \= 6-10 Hz.  
8) May want to retrieve orig EMG for more resolution when very zoomed in.  
9) Define stages  
   1) Wake: EMG\> and Delta\<; NREM: EMG\< and Delta\>; REM: EMG\< and T:D\>  
   2) Set levels by eye, while looking at known wake/nrem/rem bouts, and following major criteria and sequence restrictions above.  
   3) Set levels generously, to make it more likely that each stage will be caught correctly. Note the way the script works: wake gets classified first, then nrem, then rem, so rem toughest to be caught by automatic scoring.  
   4) Unclassified portion 50%; ambiguous portion 10%  
10) Edit sleep states  
    1) Zoom in so that you can at least make out individual 5 s epochs.  
    2) Multi epoch mode \- fill in correct stage between two marker cursors.  
    3) \* in order to make sure that stage channel is saved, you must quit the OSD programme, not just close the file\! \*  
11) Can plot power (FFT 1024\) and tables at end.  
12) Back and Quit script, or else stage channel is not saved\!\!  
13) Export as .mat file, All channels, use source name and source channel name, layout options: waveform \*and times\*
'<ADbasic Header, Headerversion 001.001>
' Process_Number                 = 1
' Initial_Processdelay           = 1000
' Eventsource                    = Timer
' Control_long_Delays_for_Stop   = No
' Priority                       = High
' Version                        = 1
' ADbasic_Version                = 6.3.0
' Optimize                       = Yes
' Optimize_Level                 = 1
' Stacksize                      = 1000
' Info_Last_Save                 = TUD209470  TUD209470\LocalAdmin
'<Header End>
Import Math.li9
#define max_array_size 100000

dim data_1[max_array_size] as long  ' input 1 data
dim data_2[max_array_size] as long  ' input 2 data
dim data_3[max_array_size] as float  ' output data

dim inc1, inc2, to_finish, index as long
dim sum as float

init:
  ' Initialized in python:
  ' - par_1: input 1 (only sometimes)
  ' - par_2: input 2 (only sometimes)
  ' - par_3: averaging of output
  ' - par_4: max input 1 voltage before amplification in Volts
  ' - par_5: max input 2 voltage before amplification in Volts
  ' - par_7: index of the data arrays
  ' - par_8: maximum number of eventser
  sum = 0  ' sum for averaging output
  par_6 = 0  ' counter for average
  par_9 = 0  ' events counter
  to_finish = 0  'input sweeps left to finish
  
  if (par_4  = -1) then
    inc1 = 0
  else
    to_finish = to_finish + 1
    if (par_1 > par_4) then
      inc1 = -1
    else
      inc1 = 1
    endif
  endif
  if (par_5  = -1) then
    inc2 = 0
  else
    to_finish = to_finish + 1
    if (par_2 > par_5) then
      inc2 = -1
    else
      inc2 = 1
    endif
  endif
  
  ' ensure if both inputs are set to be const we do not stop the process
  if (par_4 + par_5 = -2) then to_finish = 1
  
  
event:
  dac(1, par_1)
  dac(2, par_2)
  inc(par_6)
  sum = sum + adc(1)
  if (par_6 >= par_3) then
    inc(par_9)
    fpar_1 = round(sum/par_6)
    par_6 = 0
    sum = 0
    ' fill arrays
    index = mod(par_7, max_array_size)
    data_1[index] = par_1
    data_2[index] = par_2
    data_3[index] = fpar_1
    inc(par_7)
    if (inc1 = 0) then
    else
      if (par_1 = par_4) then
        to_finish = to_finish -1
      else
        par_1 = par_1 + inc1
      endif
    endif
    if (inc2 = 0) then
    else
      if (par_2 = par_5) then
        to_finish = to_finish -1
      else
        par_2 = par_2 + inc2
      endif
    endif
    if ((to_finish < 1) or (par_9 = par_8)) then stop_process(1)
  endif

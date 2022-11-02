# LAB SCRIPTS

These are the scripts I wrote to take measurements during my Master thesis project.
`general.bas` is a general purpose ADBasic script to run measurements using the ADwin, in particular, it sets the inputs and collects the real time data in temporary arrays.
The Python script controls every other part of the measurement, including:

- Setting the input and output of the measurements, such as gain settings and inputs range;
- Setting the temperature, by interfacing with the temperature controller, including waiting for the appropriate stabilization time;
- Starting/Stopping the ADwin process;
- Periodically reading the data from the ADwin memory, plot the data in real time, and save it in \texttt{csv} format at the end of each measurements;
- Allows for setting multiple consecutive measurement, e.g, at different temperature or gate voltage, automating the process.

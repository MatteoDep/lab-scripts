# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 17:11:40 2020

@author: LocalAdmin
"""

from pymeasure.instruments import Instrument
from pymeasure.instruments.validators import strict_discrete_set
import time

class LakeShore331(Instrument):
    """ Represents the Lake Shore 331 Temperature Controller and provides
    a high-level interface for interacting with the instrument.

    .. code-block:: python

        controller = LakeShore331("GPIB::1")

        print(controller.setpoint_1)        # Print the current setpoint for loop 1
        controller.setpoint_1 = 50          # Change the setpoint to 50 K
        controller.heater_range = 'low'     # Change the heater range to Low
        controller.wait_for_temperature()   # Wait for the temperature to stabilize
        print(controller.temperature_A)     # Print the temperature at sensor A

    """

    temperature_A = Instrument.measurement(
        "KRDG? A",
        """ Reads the temperature of the sensor A in Kelvin. """
    )
    temperature_B = Instrument.measurement(
        "KRDG? B",
        """ Reads the temperature of the sensor B in Kelvin. """
    )
    setpoint_1 = Instrument.control(
        "SETP? 1", "SETP 1, %g",
        """ A floating point property that controls the setpoint temperature
        in Kelvin for Loop 1. """
    )
    setpoint_2 = Instrument.control(
        "SETP? 2", "SETP 2, %g",
        """ A floating point property that controls the setpoint temperature
        in Kelvin for Loop 2. """
    )
    heater_range = Instrument.control(
        "RANGE?", "RANGE %d",
        """ A string property that controls the heater range, which
        can take the values: off, low, medium, and high. These values
        correlate to 0, 0.5, 5 and 50 W respectively. """,
        validator=strict_discrete_set,
        values={'off':0, 'low':1, 'medium':2, 'high':3},
        map_values=True
    )

    def __init__(self, adapter, **kwargs):
        super(LakeShore331, self).__init__(
            adapter,
            "Lake Shore 331 Temperature Controller",
            **kwargs
        )

    def disable_heater(self):
        """ Turns the :attr:`~.heater_range` to :code:`off` to disable the heater. """
        self.heater_range = 'off'

    def wait_for_temperature(self, accuracy=0.5, 
            interval=1, sensor='A', setpoint=1, timeout=28800,
            should_stop=lambda: False, debug=False):
        """ Blocks the program, waiting for the temperature to reach the setpoint
        within the accuracy (%), checking this each interval time in seconds.

        :param accuracy: An acceptable percentage deviation between the 
                         setpoint and temperature
        :param interval: A time in seconds that controls the refresh rate
        :param sensor: The desired sensor to read, either A or B
        :param setpoint: The desired setpoint loop to read, either 1 or 2
        :param timeout: A timeout in seconds after which an exception is raised
        :param should_stop: A function that returns True if waiting should stop, by
                            default this always returns False
        :param debug: Print values for debugging.
        """
        temperature_name = 'temperature_%s' % sensor
        setpoint_name = 'setpoint_%d' % setpoint
        # Only get the setpoint once, assuming it does not change
        setpoint_value = getattr(self, setpoint_name)
        def percent_difference(temperature):
            res = 100*abs(temperature - setpoint_value)/setpoint_value
            if debug:
                print('temp:', temperature)
                print('setpoint:', setpoint_value)
                print(f'diff: {res}%')
            return res
            
        t = time.time()
        while percent_difference(getattr(self, temperature_name)) > accuracy:
            time.sleep(interval)
            if (time.time()-t) > timeout:
                raise TimeoutError((
                    "Timeout occurred after waiting %g seconds for "
                    "the LakeShore 331 temperature to reach %g K."
                ) % (timeout, setpoint))
            if should_stop():
                return 
        
        


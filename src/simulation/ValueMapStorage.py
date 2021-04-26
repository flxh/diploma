from simulation.Storage import Storage
from scipy.interpolate import LinearNDInterpolator, CubicSpline
from scipy.io import loadmat
import numpy as np

#vm = Value Map = Kennfeld
# *_ac_power power drawn from/pushed into the converter
# *_dc_power electrical power drawn from/pushed into the battery
# *_battery_power power that is expended by the stored chemical energy of the battery


class ValueMapStorage(Storage):
    def __init__(self, capacity, soc_initial, dt_step, converter_vm_file, battery_vm_file):
        super().__init__(capacity=capacity, soc_initial=soc_initial, dt_step=dt_step)

        self.f_converter_efficiency = self._build_converter_efficiency_function(converter_vm_file)
        self.f_battery_loss = self.build_battery_loss_function(battery_vm_file)

    def build_battery_loss_function(self, battery_vmfile):
        battery_vm_mat = loadmat(battery_vmfile)
        battery_vm = battery_vm_mat['BatterieKennfeld']
        return LinearNDInterpolator(battery_vm[:,0:2], battery_vm[:,2])

    def _build_converter_efficiency_function(self, converter_vm_file):
        converter_vmdata = np.genfromtxt(converter_vm_file, delimiter=';')
        return CubicSpline(converter_vmdata[:,0], converter_vmdata[:,1], extrapolate=False)

    def step(self):
        eta_converter = self.f_converter_efficiency(np.abs(self.scheduled_power_ac))
        battery_loss = None
        # charge
        if self.scheduled_power_ac > 0:
            scheduled_dc_power = self.scheduled_power_ac * eta_converter
            battery_loss = self.f_battery_loss(scheduled_dc_power, self.soc())
            scheduled_battery_power = scheduled_dc_power - battery_loss
            # charge with scheduled power or charge left capacity
            actual_battery_power = min(scheduled_battery_power, (0.99*self.capacity - self.stored_energy)/self.dt_step)
            # prevent discharging below SOC 0.01 when battery loss > charging power
            actual_battery_power = max(actual_battery_power, (0.01*self.capacity-self.stored_energy)/self.dt_step)
            actual_dc_power = scheduled_dc_power * actual_battery_power / scheduled_battery_power
            actual_power_ac = actual_dc_power / eta_converter
        # discharge
        else:
            scheduled_dc_power = self.scheduled_power_ac / eta_converter
            battery_loss = self.f_battery_loss(scheduled_dc_power, self.soc())
            scheduled_battery_power = scheduled_dc_power - battery_loss

            # max function because power is negative when discharging
            actual_battery_power = max(scheduled_battery_power, (0.01*self.capacity-self.stored_energy)/self.dt_step)

            actual_dc_power = scheduled_dc_power * actual_battery_power / scheduled_battery_power
            actual_power_ac = actual_dc_power * eta_converter

        if np.isnan(actual_battery_power):
            raise ValueError(f'Battery-Power must not be Nan: P_AC_S: {self.scheduled_power_ac}, '
                             f'P_DC_S: {scheduled_dc_power}, P_B_S: {scheduled_battery_power}, '
                             f'P_B_A: {actual_battery_power}, P_DC_A: {actual_dc_power}, '
                             f'P_AC_A: {actual_power_ac}, P_L: {battery_loss}, eta: {eta_converter}, SOC: {self.soc()}')

        self.stored_energy += actual_battery_power * self.dt_step
        return actual_power_ac



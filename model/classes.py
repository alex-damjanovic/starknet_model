from datetime import timedelta
import math


class Entity:

    def __init__(self,
                 name: str,
                 delay: int = 0,
                 frequency: int = 1,
                 tge_unlock_percentage: float = 0.0):
        self.name = name
        self.balance = 0.0
        self.total_allocation = 0.0
        self.locked = 0.0
        self.vesting_started = False
        self.vesting_length_weeks = 0
        self.remaining_weeks_vesting = 0
        self.percentage_unlocked_per_week = 0.0
        self.cliff_length_weeks = 0
        self.remaining_weeks_cliff = 0
        self.delay = delay
        self.frequency = frequency
        self.tge_unlock_percentage = tge_unlock_percentage

    def update_delay(self, delay: int):
        self.delay = delay

    def update_frequency(self, frequency: int):
        self.frequency = frequency

    def start_vesting(self):
        self.vesting_started = True
        # Apply the TGE unlock percentage immediately using the attribute
        tge_unlock_amount = self.total_allocation * (
            self.tge_unlock_percentage / 100)
        self.balance += tge_unlock_amount
        self.locked = self.total_allocation - tge_unlock_amount

        # Adjust vesting start by the delay
        self.remaining_weeks_vesting = self.vesting_length_weeks + self.delay
        self.remaining_weeks_cliff = self.cliff_length_weeks + self.delay

        if self.vesting_length_weeks == 0 and self.cliff_length_weeks == 0:
            self.balance = self.total_allocation
            self.locked = 0.0
        else:
            self._calculate_cliff_unlock_percentage()

    def update_total_allocation(self, total_allocation: float):
        self.total_allocation = total_allocation
        self.locked = total_allocation

    def update_vesting_length_months(self, vesting_length: int):
        self.vesting_length_weeks = round(vesting_length * 4.34524)
        self._calculate_cliff_unlock_percentage()

    def update_cliff_length_weeks(self, cliff_length_weeks: int):
        self.cliff_length_weeks = cliff_length_weeks
        # Adjust cliff start by the delay
        self.remaining_weeks_cliff = cliff_length_weeks + self.delay
        self._calculate_cliff_unlock_percentage()

    def _calculate_cliff_unlock_percentage(self):
        if self.vesting_length_weeks > 0:
            self.cliff_unlock_percentage = (self.cliff_length_weeks /
                                            self.vesting_length_weeks) * 100
            effective_vesting_weeks = max(
                (self.vesting_length_weeks - self.cliff_length_weeks) /
                self.frequency, 1)
            self.percentage_unlocked_per_week = 100.0 / effective_vesting_weeks
        else:
            self.cliff_unlock_percentage = 0.0
            self.percentage_unlocked_per_week = 0.0

    def simulate_week_passage(self):
        if not self.vesting_started or self.delay > 0:
            # If vesting hasn't started or we're within the delay period, decrement delay and do nothing else
            if self.delay > 0:
                self.delay -= 1
            return

        if self.remaining_weeks_cliff > 0:
            self.remaining_weeks_cliff -= 1
            if self.remaining_weeks_cliff == 0:
                cliff_unlock_amount = self.total_allocation * (
                    self.cliff_unlock_percentage / 100)
                self.balance += cliff_unlock_amount
                self.locked -= cliff_unlock_amount
        elif self.remaining_weeks_vesting > 0:
            if self.remaining_weeks_vesting % self.frequency == 0:
                weekly_unlock_amount = self.total_allocation * (
                    self.percentage_unlocked_per_week / 100.0) * self.frequency
                new_balance = self.balance + weekly_unlock_amount
                new_locked = self.locked - weekly_unlock_amount
                if new_balance > self.total_allocation:
                    new_balance = self.total_allocation
                    new_locked = 0.0
                self.balance = new_balance
                self.locked = new_locked
            self.remaining_weeks_vesting -= 1

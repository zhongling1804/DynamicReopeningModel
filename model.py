import logging
import numpy as np
'''
States:
    Defination: Matrix(State, Time)
    Initial State: Matrix(:, 0)
'''
STATE_INDEX_NUM = 8
DEATH_INDEX, RECOVERED_INDEX, INFECTED_SYMPTOMATIC_SEVERE_INDEX, INFECTED_SYMPTOMATIC_MILD_INDEX, INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX, INFECTED_ASYMPTOMATIC_QUARANTINE_INDEX, NOT_INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX, NOT_INFECTED_ASYMPTOMATIC_QUARANTINE_INDEX = list(range(STATE_INDEX_NUM))
'''
Choices:
    Defination: Matrix(Policy, Time - 1)
    Choice Between State 0 and 1: Matrix(:, 0)
'''
CHOICE_INDEX_NUM = 9
# Local transmission: number of people a person meets per day
LOCAL_TRANSMISSION_INDEX, NUMBER_OF_TRAVELLER_FROM_MEDIUM_RISK_COUNTRY_INDEX, NUMBER_OF_TRAVELLER_FROM_HIGH_RISK_COUNTRY_INDEX, TRAVELLER_RATIO_INDEX, DAYS_OF_QUARANTINE_INDEX, TEST_ALL_QUARANTINED_TRAVELLERS_INDEX, TEST_EFFICIENCY_INDEX, TEST_NUMBER_INDEX, TRACE_INDEX = list(range(CHOICE_INDEX_NUM))
# the following parameters usually don't change.
INFECTED_FATALITY_RATE = 0.006
MILD_CASE_FRAC = 0.8
IA_FRAC = 0.4
AVERAGE_INCUBATION_DAYS = 5.2
MEDIAN_R0 = 2.79
AVERAGE_STAY_IN_HOSPITAL = 14
TRANSMISSION_RATE = MEDIAN_R0 / AVERAGE_INCUBATION_DAYS
FIRST_LOW_LOCAL_TRANSMISSION = 0.25
FIRST_MED_LOCAL_TRANSMISSION = 0.7
FIRST_HIGH_LOCAL_TRANSMISSION = 1
FIRST_PLANE_LOCAL_TRANSMISSION = FIRST_LOW_LOCAL_TRANSMISSION + FIRST_MED_LOCAL_TRANSMISSION + FIRST_HIGH_LOCAL_TRANSMISSION
TEST_EFFICIENCY = 0.8
TEST_NUMBER = 2
TRAVELLER_RATIO = 1
TRACE_LOW = 1
TRACE_MEDIUM = 0.5
TRACE_HIGH = 0.5
'''
TRANSITION_MATRIX_BASELINE
    should be Matrix(State_len, State_len)
    i.e. Matrix(To, From) in (TRANSITION_MATRIX_BASELINE * State)
'''
TRANSITION_MATRIX_BASELINE = np.zeros((STATE_INDEX_NUM, STATE_INDEX_NUM))
# DEATH_INDEX
TRANSITION_MATRIX_BASELINE[DEATH_INDEX, DEATH_INDEX] = 1
# RECOVERED_INDEX
TRANSITION_MATRIX_BASELINE[RECOVERED_INDEX, RECOVERED_INDEX] = 1
RECOVERY_RATE = 1 / AVERAGE_STAY_IN_HOSPITAL
DIAGNOSIS_RATE = 1 - np.power(IA_FRAC, 1 / AVERAGE_INCUBATION_DAYS)
IA_RECOVERY_RATE = 1 - np.power(1 - IA_FRAC, 1 / AVERAGE_STAY_IN_HOSPITAL)
MILD_MORTALITY_RATE = 0
SEVERE_MORTALITY_RATE = INFECTED_FATALITY_RATE / ((1 - IA_FRAC) * (1 - MILD_CASE_FRAC))
SEVERE_PATIENT_RATE = 1 - np.power(MILD_CASE_FRAC, 1 / AVERAGE_STAY_IN_HOSPITAL)
# INFECTED_SYMPTOMATIC_MILD_INDEX
TRANSITION_MATRIX_BASELINE[INFECTED_SYMPTOMATIC_MILD_INDEX, INFECTED_SYMPTOMATIC_MILD_INDEX] = 1 - RECOVERY_RATE - RECOVERY_RATE * MILD_MORTALITY_RATE - SEVERE_PATIENT_RATE
TRANSITION_MATRIX_BASELINE[RECOVERED_INDEX, INFECTED_SYMPTOMATIC_MILD_INDEX] = RECOVERY_RATE
TRANSITION_MATRIX_BASELINE[DEATH_INDEX, INFECTED_SYMPTOMATIC_MILD_INDEX] = RECOVERY_RATE * MILD_MORTALITY_RATE
TRANSITION_MATRIX_BASELINE[INFECTED_SYMPTOMATIC_SEVERE_INDEX, INFECTED_SYMPTOMATIC_MILD_INDEX] = SEVERE_PATIENT_RATE
# INFECTED_SYMPTOMATIC_SEVERE_INDEX
TRANSITION_MATRIX_BASELINE[INFECTED_SYMPTOMATIC_SEVERE_INDEX, INFECTED_SYMPTOMATIC_SEVERE_INDEX] = 1 - RECOVERY_RATE - RECOVERY_RATE * SEVERE_MORTALITY_RATE
TRANSITION_MATRIX_BASELINE[RECOVERED_INDEX, INFECTED_SYMPTOMATIC_SEVERE_INDEX] = RECOVERY_RATE
TRANSITION_MATRIX_BASELINE[DEATH_INDEX, INFECTED_SYMPTOMATIC_SEVERE_INDEX] = RECOVERY_RATE * SEVERE_MORTALITY_RATE
# INFECTED_ASYMPTOMATIC_QUARANTINE_INDEX
TRANSITION_MATRIX_BASELINE[RECOVERED_INDEX, INFECTED_ASYMPTOMATIC_QUARANTINE_INDEX] = IA_RECOVERY_RATE
TRANSITION_MATRIX_BASELINE[INFECTED_SYMPTOMATIC_MILD_INDEX, INFECTED_ASYMPTOMATIC_QUARANTINE_INDEX] = DIAGNOSIS_RATE
TRANSITION_MATRIX_BASELINE[INFECTED_ASYMPTOMATIC_QUARANTINE_INDEX, INFECTED_ASYMPTOMATIC_QUARANTINE_INDEX] = 1 - DIAGNOSIS_RATE - IA_RECOVERY_RATE
# INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX
TRANSITION_MATRIX_BASELINE[RECOVERED_INDEX, INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX] = IA_RECOVERY_RATE
TRANSITION_MATRIX_BASELINE[INFECTED_SYMPTOMATIC_MILD_INDEX, INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX] = DIAGNOSIS_RATE
TRANSITION_MATRIX_BASELINE[INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX, INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX] = 1 - DIAGNOSIS_RATE - IA_RECOVERY_RATE
# NOT_INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX
TRANSITION_MATRIX_BASELINE[INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX, NOT_INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX] = 1  # replaced by LT later.
TRANSITION_MATRIX_BASELINE[NOT_INFECTED_ASYMPTOMATIC_QUARANTINE_INDEX, NOT_INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX] = 0  # should be tiny, e.g.
# NOT_INFECTED_ASYMPTOMATIC_QUARANTINE_INDEX
TRANSITION_MATRIX_BASELINE[NOT_INFECTED_ASYMPTOMATIC_QUARANTINE_INDEX, NOT_INFECTED_ASYMPTOMATIC_QUARANTINE_INDEX] = 1
# print(np.sum(TRANSITION_MATRIX_BASELINE[:,4]))
for idx in range(STATE_INDEX_NUM):
    assert sum(TRANSITION_MATRIX_BASELINE[:, idx]) == 1. or logging.error("[TRANSITION_MATRIX_BASELINE] %d", idx)
# print(TRANSITION_MATRIX_BASELINE)
'''
QUARANTINE_MATRIX_BASELINE
This is the matrix for travellers during quarantine.
'''
QUARANTINE_MATRIX_BASELINE = TRANSITION_MATRIX_BASELINE.copy()
# NOT_INFECTED_ASYMPTOMATIC_QUARANTINE_INDEX
QUARANTINE_MATRIX_BASELINE[NOT_INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX, NOT_INFECTED_ASYMPTOMATIC_QUARANTINE_INDEX] = 0
QUARANTINE_MATRIX_BASELINE[NOT_INFECTED_ASYMPTOMATIC_QUARANTINE_INDEX, NOT_INFECTED_ASYMPTOMATIC_QUARANTINE_INDEX] = 1
# NOT_INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX
QUARANTINE_MATRIX_BASELINE[:, NOT_INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX] = 0
QUARANTINE_MATRIX_BASELINE[NOT_INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX, NOT_INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX] = 1
# INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX
QUARANTINE_MATRIX_BASELINE[:, INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX] = 0
QUARANTINE_MATRIX_BASELINE[INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX, INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX] = 1
for idx in range(STATE_INDEX_NUM):
    assert sum(QUARANTINE_MATRIX_BASELINE[:, idx]) == 1. or logging.error("[QUARANTINE_MATRIX_BASELINE] %d", idx)
'''
Crowdout represents the following ratio:
crowdout = Pr(infected,asymptomatic|on the plane)/Pr(infected,asymptomatic|in the departure country population and are eligible to board the plane).
We think this factor should increase as Pr(infected,asymptomatic|in the departure country) increase, and should be greater than 1 as long as the plane goes from a higher risk country to a lower risk country.
'''


class LowRiskCountry(object):
    # the parameter of 0.098 comes from the formula for quarantine function h() = contact tracing * sqrt (diagnosis rate * transmission rate * social distancing)
    INIT_STATE = (0.000003, 0.00006, 2.4e-7, 0.6e-7, 4.5e-7 - (2.4e-7 + 0.6e-7) / 14 * 5.2 * (TRACE_LOW * np.power((1 / AVERAGE_INCUBATION_DAYS) * (MEDIAN_R0 / AVERAGE_INCUBATION_DAYS) * FIRST_LOW_LOCAL_TRANSMISSION), 1 / 2), (2.4e-7 + 0.6e-7) / 14 * 5.2 * (TRACE_LOW * np.power((1 / AVERAGE_INCUBATION_DAYS) * (MEDIAN_R0 / AVERAGE_INCUBATION_DAYS) * FIRST_LOW_LOCAL_TRANSMISSION), 1 / 2), 0.99993624, 0)
    ID = 1

    def __init__(self, max_time=300):
        # initialize the matrices with duration of policy = max_time.
        # state (distribution of people across various states) of low risk country in each time period
        self._states = np.zeros((STATE_INDEX_NUM, max_time))
        self._leave_here = np.zeros((STATE_INDEX_NUM, max_time))
        # policy of low risk country in each time period
        self._choices = np.zeros((CHOICE_INDEX_NUM, max_time))  # last choice is not used
        self.INIT_STATE = np.asarray(self.INIT_STATE)
        assert self.INIT_STATE.shape[0] == STATE_INDEX_NUM
        self._states[:, 0] = self.INIT_STATE
        # make sure people sum to 1 in the initial state
        assert np.isclose(np.sum(self._states[:, 0]), 1.) or logging.error("INIT_STATE %d -> %f", self.ID, np.sum(self._states[:, 0]))
        self.setChoicesBaseline()
        self._medium_risk_table = None
        self._high_risk_table = None

    def setChoicesBaseline(self):
        self._choices[LOCAL_TRANSMISSION_INDEX, :] = FIRST_LOW_LOCAL_TRANSMISSION
        self._choices[DAYS_OF_QUARANTINE_INDEX, :] = 7
        # China 2018 data: daily traveler inflow is 0.0002 of local population
        # China 2020 March 28 onward data: daily traveler inflow is 0.0000167
        self._choices[NUMBER_OF_TRAVELLER_FROM_MEDIUM_RISK_COUNTRY_INDEX, :] = 0.00005
        self._choices[NUMBER_OF_TRAVELLER_FROM_HIGH_RISK_COUNTRY_INDEX, :] = 0.00005
        self._choices[TEST_EFFICIENCY_INDEX, :] = TEST_EFFICIENCY
        self._choices[TEST_NUMBER_INDEX, :] = TEST_NUMBER
        self._choices[TRAVELLER_RATIO_INDEX, :] = TRAVELLER_RATIO
        self._choices[TRACE_INDEX, :] = TRACE_LOW
        self._is_low = True
        self.preset_total_new_case = 1
        logging.warning("COUNTRY,TIME,DEATH_INDEX,RECOVERED_INDEX,INFECTED_SYMPTOMATIC_SEVERE_INDEX,INFECTED_SYMPTOMATIC_MILD_INDEX,INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX,INFECTED_ASYMPTOMATIC_QUARANTINE_INDEX,NOT_INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX,NOT_INFECTED_ASYMPTOMATIC_QUARANTINE_INDEX")

    def getState(self, t):
        current_state = self._states[:, t].copy()
        # assert sum(current_state) + 1e-10 >= 1. or print(current_state)
        return current_state

    def getChoicesForDraw(self, time_we_want=200):
        return self._choices[:, :time_we_want]

    def getStatesForDraw(self, time_we_want=200):
        return self._states[:, :time_we_want]

    def getNewCases(self, time_we_want=200):
        # new cases = (death + recovered + infected,symptomatic) at time t - (death+recovered+infected,symptomatic) at time (t-1)
        all_cases = self._states[DEATH_INDEX, :time_we_want] + self._states[RECOVERED_INDEX, :time_we_want] + self._states[INFECTED_SYMPTOMATIC_MILD_INDEX, :time_we_want] + self._states[INFECTED_SYMPTOMATIC_SEVERE_INDEX, :time_we_want]
        all_cases = np.diff(all_cases).flatten()
        assert len(all_cases) == time_we_want - 1
        return all_cases

    def getNewSevereCases(self, time_we_want=200):
        # new cases = (death + recovered + infected,symptomatic) at time t - (death+recovered+infected,symptomatic) at time (t-1)
        all_cases = self._states[DEATH_INDEX, :time_we_want] + self._states[RECOVERED_INDEX, :time_we_want] + self._states[INFECTED_SYMPTOMATIC_SEVERE_INDEX, :time_we_want]
        all_cases = np.diff(all_cases).flatten()
        assert len(all_cases) == time_we_want - 1
        return all_cases

    def crowdOut(self, depart_state, num_traveller, t, destin_state):
        depart_state = depart_state.copy()
        assert self._is_low

        boarding_pre_crowdout = depart_state
        # Eligible travellers: infected,symptomatic; recovered; not infected, not quarantine.
        boarding_pre_crowdout[DEATH_INDEX] = 0
        boarding_pre_crowdout[INFECTED_SYMPTOMATIC_MILD_INDEX] = 0
        boarding_pre_crowdout[INFECTED_SYMPTOMATIC_SEVERE_INDEX] = 0
        boarding_pre_crowdout[INFECTED_ASYMPTOMATIC_QUARANTINE_INDEX] = 0
        boarding_pre_crowdout[NOT_INFECTED_ASYMPTOMATIC_QUARANTINE_INDEX] = 0

        # Infected,asymptomatic people have higher incentive to fly out of the epicenter, so they crowd out other people.
        subtotal = sum(boarding_pre_crowdout)
        boarding_pre_crowdout /= subtotal
        assert np.isclose(np.sum(boarding_pre_crowdout), 1)
        if num_traveller == 0:
            return boarding_pre_crowdout

        # init post_crowdout vector
        people_on_plane = boarding_pre_crowdout

        people_on_plane[INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX] = max(0, min(1.149 * np.power(depart_state[INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX] * DIAGNOSIS_RATE, 0.5), 0.5 * depart_state[INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX]))
        if people_on_plane[INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX] == 0.5 * depart_state[INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX]:
            print("touch upperbound")

        # After infected,asymptomatic people occupy many seats on the plane, the rest of the seats are distributed between recovered and uninfected people, in the same proportion as they have in the population of the departure country.
        # nanq_new = (1-ianq_adjusted)*(nanq/(nanq+r))
        rest_ratio_for_r_and_nanq = 1 - people_on_plane[INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX]
        subtotal_of_r_and_nanq = boarding_pre_crowdout[NOT_INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX] + boarding_pre_crowdout[RECOVERED_INDEX]
        r_ratio_in_rest = boarding_pre_crowdout[RECOVERED_INDEX] / subtotal_of_r_and_nanq
        nanq_ratio_in_rest = boarding_pre_crowdout[NOT_INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX] / subtotal_of_r_and_nanq
        people_on_plane[NOT_INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX] = nanq_ratio_in_rest * rest_ratio_for_r_and_nanq
        people_on_plane[RECOVERED_INDEX] = r_ratio_in_rest * rest_ratio_for_r_and_nanq

        assert np.isclose(np.sum(people_on_plane), 1)
        return people_on_plane

    def AdvanceFromLowView(self, time_we_want=200):
        logging.warning("%d,%d,%s", self.ID, 0, ",".join([str(idx) for idx in self.INIT_STATE]))
        assert self._is_low
        self._medium_risk_table = MediumRiskCountry(self._states.shape[1])
        logging.warning("%d,%d,%s", self._medium_risk_table.ID, 0, ",".join([str(idx) for idx in self._medium_risk_table.INIT_STATE]))
        self._high_risk_table = HighRiskCountry(self._states.shape[1])
        logging.warning("%d,%d,%s", self._high_risk_table.ID, 0, ",".join([str(idx) for idx in self._high_risk_table.INIT_STATE]))
        for idx in range(time_we_want):
            self.AdvanceStateFromLowView(idx)

    def applyPolicyOnMatrix(self, current_state, lt, islocal, trace):
        assert np.isclose(np.sum(current_state), 1)
        real_used_transition_matrix = TRANSITION_MATRIX_BASELINE.copy()
        # NOT_INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX = number of people you meet each day (determined by local pandemic policy, represented by LT) * Pr(infected|not quarantined) * Pr(uninfected|not quarantined)
        infected_nq = current_state[INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX]
        healthy = current_state[NOT_INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX]
        nq_tot = infected_nq + healthy + current_state[RECOVERED_INDEX] + current_state[NOT_INFECTED_ASYMPTOMATIC_QUARANTINE_INDEX] + current_state[INFECTED_ASYMPTOMATIC_QUARANTINE_INDEX]
        real_used_transition_matrix[INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX, NOT_INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX] = min(lt * TRANSMISSION_RATE * (infected_nq / nq_tot), 0.9)
        real_used_transition_matrix[NOT_INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX, NOT_INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX] = 1 - real_used_transition_matrix[INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX, NOT_INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX]
        # INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX: some are quarantined.
        if islocal:
            # number of people a IANQ person can infect
            num_people_infect = lt * (healthy / nq_tot)
            # adjust and prep for probability calculation
            num_people_infect = min(num_people_infect, 1 / DIAGNOSIS_RATE)
            # theoretical quarantine probability
            ianq_iaq_theory = np.power(num_people_infect * DIAGNOSIS_RATE, 0.5) * trace
            # theoretical hospitalization probability
            ianq_ism_theory = real_used_transition_matrix[INFECTED_SYMPTOMATIC_MILD_INDEX, INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX]
            ianq_iss_theory = real_used_transition_matrix[INFECTED_SYMPTOMATIC_SEVERE_INDEX, INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX]
            ianq_is_theory = ianq_ism_theory + ianq_iss_theory
            if ianq_iaq_theory + ianq_is_theory < 0.99:
                real_used_transition_matrix[INFECTED_ASYMPTOMATIC_QUARANTINE_INDEX, INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX] = ianq_iaq_theory
            else:
                real_used_transition_matrix[INFECTED_ASYMPTOMATIC_QUARANTINE_INDEX, INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX] = 0.99 * (1 - ianq_is_theory)
        else:
            real_used_transition_matrix[INFECTED_ASYMPTOMATIC_QUARANTINE_INDEX, INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX] = 0
        real_used_transition_matrix[INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX, INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX] = 1 - real_used_transition_matrix[INFECTED_ASYMPTOMATIC_QUARANTINE_INDEX, INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX] - real_used_transition_matrix[RECOVERED_INDEX, INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX] - real_used_transition_matrix[INFECTED_SYMPTOMATIC_MILD_INDEX, INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX] - real_used_transition_matrix[INFECTED_SYMPTOMATIC_SEVERE_INDEX, INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX]
        assert real_used_transition_matrix[INFECTED_ASYMPTOMATIC_QUARANTINE_INDEX, INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX] >= 0
        assert real_used_transition_matrix[INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX, INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX] >= 0
        for idx in range(STATE_INDEX_NUM):
            assert np.isclose(1, np.sum(real_used_transition_matrix[:, idx])) or logging.error("[real_used_transition_matrix] %d %s", idx, str(real_used_transition_matrix[:, idx]))
        return real_used_transition_matrix

    def decideChoiceTOfLowWhenStateT(self, t, total_new_case_used_here):
        if not self._is_low:
            return
        medium_t = self._medium_risk_table.getState(t)
        high_t = self._high_risk_table.getState(t)
        if medium_t[INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX] < high_t[INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX]:
            medium_t = self.crowdOut(medium_t, self._choices[NUMBER_OF_TRAVELLER_FROM_MEDIUM_RISK_COUNTRY_INDEX, t], t, self.getState(t))
            medium_t_used_transition_matrix = self.applyPolicyOnMatrix(medium_t, FIRST_PLANE_LOCAL_TRANSMISSION, False, 0)
            medium_t = medium_t_used_transition_matrix.dot(medium_t)
            if medium_t[INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX] > 0:
                available_tm = min(total_new_case_used_here / medium_t[INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX], self._choices[NUMBER_OF_TRAVELLER_FROM_MEDIUM_RISK_COUNTRY_INDEX, t])
            else:
                available_tm = 0
            self._choices[NUMBER_OF_TRAVELLER_FROM_MEDIUM_RISK_COUNTRY_INDEX, t] = available_tm
            total_new_case_used_here -= available_tm * medium_t[INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX]
            if total_new_case_used_here == 0:
                self._choices[NUMBER_OF_TRAVELLER_FROM_HIGH_RISK_COUNTRY_INDEX, t] = 0
            else:
                high_t = self.crowdOut(high_t, self._choices[NUMBER_OF_TRAVELLER_FROM_HIGH_RISK_COUNTRY_INDEX, t], t, self.getState(t))
                high_t_used_transition_matrix = self.applyPolicyOnMatrix(high_t, FIRST_PLANE_LOCAL_TRANSMISSION, False, 0)
                high_t = high_t_used_transition_matrix.dot(high_t)
                if high_t[INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX] > 0:
                    available_th = min(total_new_case_used_here / high_t[INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX], self._choices[NUMBER_OF_TRAVELLER_FROM_HIGH_RISK_COUNTRY_INDEX, t])
                else:
                    available_th = 0
                self._choices[NUMBER_OF_TRAVELLER_FROM_HIGH_RISK_COUNTRY_INDEX, t] = available_th
        else:
            high_t = self.crowdOut(high_t, self._choices[NUMBER_OF_TRAVELLER_FROM_HIGH_RISK_COUNTRY_INDEX, t], t, self.getState(t))
            high_t_used_transition_matrix = self.applyPolicyOnMatrix(high_t, FIRST_PLANE_LOCAL_TRANSMISSION, False, 0)
            high_t = high_t_used_transition_matrix.dot(high_t)
            if high_t[INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX] > 0:
                available_th = min(total_new_case_used_here / high_t[INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX], self._choices[NUMBER_OF_TRAVELLER_FROM_HIGH_RISK_COUNTRY_INDEX, t])
            else:
                available_th = 0
            self._choices[NUMBER_OF_TRAVELLER_FROM_HIGH_RISK_COUNTRY_INDEX, t] = available_th
            total_new_case_used_here -= available_th * high_t[INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX]
            if total_new_case_used_here == 0:
                self._choices[NUMBER_OF_TRAVELLER_FROM_MEDIUM_RISK_COUNTRY_INDEX, t] = 0
            else:
                medium_t = self.crowdOut(medium_t, self._choices[NUMBER_OF_TRAVELLER_FROM_MEDIUM_RISK_COUNTRY_INDEX, t], t, self.getState(t))
                medium_t_used_transition_matrix = self.applyPolicyOnMatrix(high_t, FIRST_PLANE_LOCAL_TRANSMISSION, False, 0)
                medium_t = medium_t_used_transition_matrix.dot(medium_t)
                if medium_t[INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX] > 0:
                    available_tm = min(total_new_case_used_here / high_t[INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX], self._choices[NUMBER_OF_TRAVELLER_FROM_MEDIUM_RISK_COUNTRY_INDEX, t])
                else:
                    available_tm = 0
                self._choices[NUMBER_OF_TRAVELLER_FROM_MEDIUM_RISK_COUNTRY_INDEX, t] = available_tm

    def peopleLeaveLowFromLowView(self, t):
        if not self._is_low:
            return
        # decide if we do something to medium and high risk country
        leave_low_risk = self.getState(t)
        leave_low_risk[DEATH_INDEX] = 0
        leave_low_risk[INFECTED_SYMPTOMATIC_MILD_INDEX] = 0
        leave_low_risk[INFECTED_SYMPTOMATIC_SEVERE_INDEX] = 0
        leave_low_risk[INFECTED_ASYMPTOMATIC_QUARANTINE_INDEX] = 0
        leave_low_risk[NOT_INFECTED_ASYMPTOMATIC_QUARANTINE_INDEX] = 0
        leave_low_risk_tot = sum(leave_low_risk)
        leave_low_risk /= leave_low_risk_tot
        #   travellers from L to M and H is a random sample of local people at L. The number of travellers allowed to board the plane is the proportional to the number of travellers coming in.
        traveller_num_high = self._choices[NUMBER_OF_TRAVELLER_FROM_HIGH_RISK_COUNTRY_INDEX, t] * self._choices[TRAVELLER_RATIO_INDEX, t]
        low_people_on_plane_to_high_ratio = self.crowdOut(leave_low_risk, traveller_num_high, t, self._high_risk_table.getState(t))
        low_to_high = low_people_on_plane_to_high_ratio * traveller_num_high
        traveller_num_medium = self._choices[NUMBER_OF_TRAVELLER_FROM_MEDIUM_RISK_COUNTRY_INDEX, t] * self._choices[TRAVELLER_RATIO_INDEX, t]
        low_people_on_plane_to_medium_ratio = self.crowdOut(leave_low_risk, traveller_num_medium, t, self._medium_risk_table.getState(t))
        low_to_medium = low_people_on_plane_to_medium_ratio * traveller_num_medium

        leave_low_real_used_transition_matrix = self.applyPolicyOnMatrix(leave_low_risk, FIRST_PLANE_LOCAL_TRANSMISSION, False, 0)
        low_to_high = leave_low_real_used_transition_matrix.dot(low_to_high)
        low_to_medium = leave_low_real_used_transition_matrix.dot(low_to_medium)

        self._medium_risk_table._states[:, t + 1] += low_to_medium
        self._high_risk_table._states[:, t + 1] += low_to_high
        self._leave_here[:, t] = low_to_medium + low_to_high
        assert np.min(self._states[:, t] - self._leave_here[:, t]) >= 0

    def someoneWantToEnterLowWhenT(self, t, from_country, traveller_num, dayQ):
        if not self._is_low:
            return
        dayQ = int(dayQ)
        people_at_that_country = from_country.getState(t)
        # get plane
        people_on_plane_ratio = self.crowdOut(people_at_that_country, traveller_num, t, self.getState(t))
        from_country._leave_here[:, t] = people_on_plane_ratio * traveller_num
        # if they infected on plane?
        plane_transition_matrix = self.applyPolicyOnMatrix(people_on_plane_ratio, FIRST_PLANE_LOCAL_TRANSMISSION, False, 0)
        people_get_off_plane = plane_transition_matrix.dot(people_on_plane_ratio) * traveller_num
        assert np.isclose(sum(people_get_off_plane), sum(from_country._leave_here[:, t]))
        # quarantine for dayQ days.
        q_for_one_day = QUARANTINE_MATRIX_BASELINE.copy()
        for _ in range(dayQ):
            people_get_off_plane = q_for_one_day.dot(people_get_off_plane)
        # test for test_num times, each test's effectiveness is (1-test_false)
        test_false = 1 - self._choices[TEST_EFFICIENCY_INDEX, t]
        test_num = self._choices[TEST_NUMBER_INDEX, t]
        if test_num > 0:
            false_negative_ianq = people_get_off_plane[INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX] * np.power(test_false, test_num)
            people_get_off_plane[INFECTED_SYMPTOMATIC_MILD_INDEX] += people_get_off_plane[INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX] - false_negative_ianq
            people_get_off_plane[INFECTED_ASYMPTOMATIC_NOT_QUARANTINE_INDEX] = false_negative_ianq

        self._states[:, t + dayQ] += people_get_off_plane
        assert np.isclose(sum(people_get_off_plane), sum(from_country._leave_here[:, t])) or print(sum(people_get_off_plane), sum(from_country._leave_here[:, t]))

    def restLocalPeopleSleepWhenT(self, t):
        rest_people = self.getState(t) - self._leave_here[:, t]
        assert np.min(rest_people) >= 0
        rest_people_ratio = rest_people / sum(rest_people)
        assert np.isclose(sum(rest_people_ratio), 1)

        # apply policy
        rest_people_transition_matrix = self.applyPolicyOnMatrix(rest_people_ratio, self._choices[LOCAL_TRANSMISSION_INDEX, t], True, self._choices[TRACE_INDEX, t])
        # get next state: epidemic evolve among local people
        rest_people_wakeup_tomorrow = rest_people_transition_matrix.dot(rest_people)
        assert np.min(rest_people_wakeup_tomorrow) >= 0
        self._states[:, t + 1] += rest_people_wakeup_tomorrow

        logging.info("[AdvanceState] Time %d from %s to %s Choice %s", t, str(self.getState(t)), str(self.getState(t + 1)), str(self._choices[:, t]))
        logging.warning("%d,%d,%s", self.ID, t + 1, ",".join([str(idx) for idx in self.getState(t + 1).tolist()]))
        assert self.getState(t + 1)[DEATH_INDEX] > 0 or logging.error("death disappear ID %d t %d %s -> %s", self.ID, t, str(self.getState(t)), str(self.getState(t + 1)))

    def AdvanceStateFromLowView(self, t):
        assert self._is_low
        # ajust policy, here we know t of L,M,H
        total_new_case_used_here = self.preset_total_new_case
        if total_new_case_used_here > 0:
            old_tm = self._choices[NUMBER_OF_TRAVELLER_FROM_MEDIUM_RISK_COUNTRY_INDEX, t]
            old_th = self._choices[NUMBER_OF_TRAVELLER_FROM_HIGH_RISK_COUNTRY_INDEX, t]
            self.decideChoiceTOfLowWhenStateT(t, total_new_case_used_here)
            new_tm = self._choices[NUMBER_OF_TRAVELLER_FROM_MEDIUM_RISK_COUNTRY_INDEX, t]
            new_th = self._choices[NUMBER_OF_TRAVELLER_FROM_HIGH_RISK_COUNTRY_INDEX, t]
            assert new_tm <= old_tm
            assert new_th <= old_th

        self.peopleLeaveLowFromLowView(t)
        self.someoneWantToEnterLowWhenT(t, self._medium_risk_table, self._choices[NUMBER_OF_TRAVELLER_FROM_MEDIUM_RISK_COUNTRY_INDEX, t], self._choices[DAYS_OF_QUARANTINE_INDEX, t])
        self.someoneWantToEnterLowWhenT(t, self._high_risk_table, self._choices[NUMBER_OF_TRAVELLER_FROM_HIGH_RISK_COUNTRY_INDEX, t], self._choices[DAYS_OF_QUARANTINE_INDEX, t])

        self.restLocalPeopleSleepWhenT(t)
        self._medium_risk_table.restLocalPeopleSleepWhenT(t)
        self._high_risk_table.restLocalPeopleSleepWhenT(t)


class MediumRiskCountry(LowRiskCountry):
    INIT_STATE = (0.0006, 0.0032, 0.0021 * 0.8, 0.0021 * 0.2, 0.0032 - (0.0021) / 14 * 5.2 * (TRACE_MEDIUM * np.power((1 / AVERAGE_INCUBATION_DAYS) * (MEDIAN_R0 / AVERAGE_INCUBATION_DAYS) * FIRST_MED_LOCAL_TRANSMISSION), 1 / 2), (0.0021) / 14 * 5.2 * (TRACE_MEDIUM * np.power((1 / AVERAGE_INCUBATION_DAYS) * (MEDIAN_R0 / AVERAGE_INCUBATION_DAYS) * FIRST_MED_LOCAL_TRANSMISSION), 1 / 2), 0.993975, 0)
    ID = 2

    def setChoicesBaseline(self):
        self._choices[LOCAL_TRANSMISSION_INDEX, :] = FIRST_MED_LOCAL_TRANSMISSION
        self._choices[TRACE_INDEX, :] = TRACE_MEDIUM
        self._is_low = False
        self.preset_total_new_case = 0


class HighRiskCountry(LowRiskCountry):
    INIT_STATE = (0.0005, 0.004, 0.0071, 0.0018, 0.0134 - (0.0071 + 0.0018) / 14 * 5.2 * (TRACE_HIGH * np.power((1 / AVERAGE_INCUBATION_DAYS) * (MEDIAN_R0 / AVERAGE_INCUBATION_DAYS) * FIRST_HIGH_LOCAL_TRANSMISSION), 1 / 2), (0.0071 + 0.0018) / 14 * 5.2 * (TRACE_HIGH * np.power((1 / AVERAGE_INCUBATION_DAYS) * (MEDIAN_R0 / AVERAGE_INCUBATION_DAYS) * FIRST_HIGH_LOCAL_TRANSMISSION), 1 / 2), 0.9732, 0)
    ID = 3

    def setChoicesBaseline(self):
        self._choices[LOCAL_TRANSMISSION_INDEX, :] = FIRST_HIGH_LOCAL_TRANSMISSION
        self._choices[TRACE_INDEX, :] = TRACE_HIGH
        self._is_low = False
        self.preset_total_new_case = 0


if __name__ == '__main__':
    raise Exception("DO NOT DIRECTLY RUN HERE")

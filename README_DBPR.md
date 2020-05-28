# A dynamic model of reopening strategy amid pandemic outbreak

This is a dynamic model of reopening strategies concerning traffic between areas in heterogeneous phases of a pandemic outbreak.

## Usage

This epidemic model has multiple areas in heterogeneous phases. The areas are connected by travelers across the areas.
The model features disproportionally selected travelers and two types of government policy interventions. The internal policies include social distancing that decreases infection rate, and contact tracing that increases quarantine probability once infected. The external policies include partial or full travel restrictions that controls the number of inbound travelers, quarantine and testing plans that pick out infectious travelers before they interact with local people.

### Environment Requirement

- Python3 with Numpy

### Run Model with Default Settings

To run model with default settings, just call in Python3:

```python3
country = LowRiskCountry()
country.AdvanceFromLowView()
```

### Results Data Structure

Core data are ***States*** and ***Choices***.

States:

- Definition: 2-dim Matrix[State, Time]
- Initial State: Matrix[ : , 0]

Choices:

- Defination: 2-dim Matrix[Policy, Time], the last one (Matrix[ : , Time - 1]) is not used
- Choice Between State 0 and State 1: Matrix[ : , 0]

We recommend to get results (states and choices) using embedded functions:

```python3
# run results
country = LowRiskCountry()
country.AdvanceFromLowView()

# get results
country.getChoicesForDraw()
country.getStatesForDraw()
country.getNewCases()
```

We construct a space of 300 results to store temporary results, but only inference final results of 200 time periods by default. Please see the definition below:

```python3
# how many spaces can be used to store temporary data
LowRiskCountry(max_time=300)

# how many results we inferenced
AdvanceFromLowView(time_we_want=200)

# how many results we want and valid (should be same as AdvanceFromLowView)
getChoicesForDraw(time_we_want=200)
getStatesForDraw(time_we_want=200)
getNewCases(time_we_want=200)
```

### Using Custom Policies

For example, if we want to test the effect of a strict social distancing policy (local people's meeting rate = 0.1) accompanied by a relaxed quota for travelers (daily inbound traveler from each foreign area is 0.1% of the local population), we set:

```python3
country = LowRiskCountry()
country._choices[NUMBER_OF_TRAVELLER_FROM_MEDIUM_RISK_COUNTRY_INDEX, :] = 0.001
country._choices[NUMBER_OF_TRAVELLER_FROM_HIGH_RISK_COUNTRY_INDEX, :] = 0.001
country._choices[LOCAL_TRANSMISSION_INDEX, :] = 0.1
country.AdvanceFromLowView()
```

## Copyright and License

Copyright (c) 2020 DBPR 2020-05-08324

The scripts provided here are released under the MIT license (<http://opensource.org/licenses/mit-license.php>).


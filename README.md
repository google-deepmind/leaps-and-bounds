--------------------------------------------------------------------------------

# This is an implementation of LeapsAndBounds and Structured Procrastination for approximately optimal algorithm configuration.

Install dependencies:

```shell
pip install -r requirements.txt
```

First uncompress the measurements file:

`gzip -d measurements.dump.gz`

To simulate a run of LeapsAndBounds:

`python leapsandbounds.py`

To simulate a run of Structured Procrastination:

`python structured_procrastination.py`

Parameters for these algorithms can be supplied via the command line. For an
explanation, use `--help`.

## Implementation details

Both leapsandbounds.py and structured_procrastination.py use
simulated_environment.py, which loads the runtime measurements from
measurements.dump. measurements.dump is a pickle dump of a python dictionary
containing the runtime measurements for 972 different configurations and 20118
generated instances. The dictionary is keyed on the command line arguments of
configurations, and its values are the corresponding measurement results: a list
of runtimes in seconds, one for each instance.

## This is not an officially supported Google product.

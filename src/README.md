# rf-fingerprints
This is a repo for the tensorflow code for the project.

## How to use
### Tools
Tools contains a `create_datset.py` script that will take data in I/Q format 
and convert it into a tensorflow record containing examples. The format of each
example is as follows:
```csv
device: A string/bytestring containing an identifier for the device the data is from.
sample_num: The sample number of this example.
length: The number of samples contained
i_values: A list of all i values ordered by time recorded
q_values: A list of all q values ordered by time recorded
```
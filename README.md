# rf-fingerprints
This is a repo for the tensorflow code for the project.

## Information
###Pandas Reader
I created a new file reader using pandas in order to get a faster read. I have not bench-marked it, but it runs quite a bit faster than the current loadtext method you are currently using. It takes raw csvs from a data and converts them into a (configurable) number of tfrecords shards. It creates one record containing a random subset of the data for testing, and a number of configurable sized shards for training in order to better help with data loading/shuffling. It can also be run as a shell script, like so: `python pandas_reader.py [-s SHARDSIZE] [-t TESTSIZE] in_root out_root`
###Data Loader
This is the main input mechanism. It is a class encapsulating a queue that reads tfrecords from a specified folder. It supports a configurable number of epochs, batch size, random cropping of data, downsampling, and noise. You can look at the code/docstring for more information. The nice thing about these queues is we can easily spin them off on as many threads as your computer can comfortably support in order to really speed up the pipeline.
###LSTM Model
This is the model itself. It is essentially the same as your model. but made more contained and configurable. It is constructed with a queue (generally retrieved from a data_loaders get_samples method) as well a configurable time step, learning rate, number of hidden units, number of classes, and whether or not to use a bidirectional or unidirectional RNN. I will be adding the ability to swap out LSTM cells for other RNN cells at some point in the future. You can train it by simply calling the train method and passing in the session. Additionally, it supports feeding with a feed_dict, so if you have custom data you want to predict/test on it can handle that as well. 
###Tensorboard
The LSTM Model is set up to produce summaries for tensorboard. The included runner.py demonstrates basic usage of the whole pipeline, as well as writing the data to tensorboard. The train and test methods of the model both return summaries for that pass, which can be added to summary writers for tensorboard. Attached is an image showing the results of running 9/23/2017's data using the example runner.py for 100,000 iterations.
# ALife - 2018
## The effects of morphology and fitness on catastrophic interference

This is the code used in the above titled paper for conducting the experiments and generating the results.

Code was run using <strong>Anaconda 5.1</strong> with <strong>Python 3.6</strong> and has as a dependecny <strong><a href=https://ccappelle.github.io/pyrosim/index.html>Pyrosim</a></strong> and <strong>tqdm</strong> (pip install tqdm)



<ul>To run the code navigate into 2 or 4 Env folder, then pick the folder for robot treatment
  <li>Baseline = Legged Robot</li>
  <li>AllWheels = Wheeled Robot</li>
  <li>WheelsAndJoints = Whegged Robot</li>
</ul>



<ul>In that folder run <strong>python main.py arg1 arg2</strong>
  <li>arg1 = the fitness function as a string, one of "+" || "*" || "min"</li>
  <li>arg2 = the random seed as an int</li>
</ul>



An example would be:

cd Morphology\ 2\ Env/WheelsAndJoints/

python main.py "min" 0

# the stroop model

this repo replicates 
<a href="https://www.ncbi.nlm.nih.gov/pubmed/2200075">cohen et al (1990)</a>
w/ 
<a href="https://github.com/PrincetonUniversity/PsyNeuLink">psyneulink</a> 
composition 
(tested on version==0.5.2.1),
this is developed for the course NEU|PSY330 
<a href="https://registrar.princeton.edu/course-offerings/course-details?courseid=005628&term=1194">Computational Modeling of Psychological Function</a>, Spring 2019 

### doc 

here're some <a href="https://docs.google.com/presentation/d/1uG9LVT5susIOUvRCeg8qzzZNwNnwRbGPTiRGcjAyqGU/edit?usp=sharing">slides</a> 
i used for the lab

- `stroop_model.py`: the definition of the stroop model
- `stroop_stimulus.py`: the definition of the stroop task; I also uploaded this helper script this as a package, so you can `pip install stroop`. Here's its <a href="https://github.com/qihongl/stroop-stimuli">repo</a>
- `run_exp_*.py`: run some experiment, where `*` is the name of the experiemnt
- `show_*.py`: analyze and plot the data 
- `stroop-feedforward.ipynb` and `stroop-linear.ipynb`: two simplifications of the full stroop model, for teaching purpose


### the model

here's the architecture... 

<img src="https://github.com/qihongl/stroop-pnlcomp/blob/master/imgs/STROOP-model.png" alt="model" height=300px>

### results

here's the main result, which qualitatively replicate fig 5 from cohen et al (1990). in general...
- color naming is slower than word reading
- color doesn't affect word reading but word affect color naming a lot. for example, color naming *red green* as "red" is much slower than word reading *red green* as "green", where *red green* = the word green painted in red. 
- magnitude(interference) > magnitude(facilitation)

<img src="https://github.com/qihongl/stroop-pnlcomp/blob/master/imgs/stroop.png" alt="rt" height=350px>


here's another way to plot the data. these are kernel density estimates of the reaction time distributions.  

<img src="https://github.com/qihongl/stroop-pnlcomp/blob/master/imgs/rt_kde.png" alt="rt kde" height=450px>

here's the effect of demand 
- the left panel corresponds to fig 13 a in Cohen et al. 1990
- note that the right panel is not plotting fig 13 b in the paper -- we don't have a shape naming condition here 

<img src="https://github.com/qihongl/stroop-cohen-etal-1990/blob/master/imgs/demand.png" alt="demand" height=300px>


and here's the SOA effect - figure 7 in Cohen et al 1990: 

<img src="https://github.com/qihongl/stroop-pnlcomp/blob/master/imgs/soa.png" alt="soa" height=300px>


decision energy, idea from Shenhav et al. 2013: 

<img src="https://github.com/qihongl/stroop-cohen-etal-1990/blob/master/imgs/dec_act.png" alt="dec_eng" height=250px>




References: 

[1] Cohen, J. D., Dunbar, K., & McClelland, J. L. (1990). On the control of automatic processes: a parallel distributed processing account of the Stroop effect. Psychological Review, 97(3), 332–361. Retrieved from https://www.ncbi.nlm.nih.gov/pubmed/2200075

[2] Shenhav, A., Botvinick, M. M., & Cohen, J. D. (2013). The expected value of control: an integrative theory of anterior cingulate cortex function. Neuron, 79(2), 217–240. https://doi.org/10.1016/j.neuron.2013.07.007

[3] the code is based on all stroop model scripts I can find from psyneulink example scripts 
<a href="https://github.com/PrincetonUniversity/PsyNeuLink/tree/master/Scripts">here</a>, 
and most importantly,
<a href="https://github.com/PrincetonUniversity/PsyNeuLink/blob/master/Scripts/Examples/Stroop%20Basic.py">this</a> 
and 
<a href="https://github.com/PrincetonUniversity/PsyNeuLink/blob/master/Scripts/Laura%20Stroop.py">this</a>

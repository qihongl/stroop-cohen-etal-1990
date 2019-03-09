# stroop-pnlcomp, for neu/psy 330 spring 2019 

this repo replicates cohen et al (1990) w/ psyneulink composition (version==0.5.2.1)

here's the model... 

<img src="https://github.com/qihongl/stroop-pnlcomp/blob/master/imgs/STROOP-model.png" alt="model" height=300px>


here's the main result, meant to qualitatively replicate fig 5 from cohen et al (1990). so in general, color naming is slower than word reading; color naming *red green* as "red" is much slower than word reading *red green* as "green", where *red green* = the word green painted in red. 

<img src="https://github.com/qihongl/stroop-pnlcomp/blob/master/imgs/stroop_0.2.png" alt="model" height=350px>


here's another way to plot the data. these are kernel density estimates of the reaction time distributions.  

<img src="https://github.com/qihongl/stroop-pnlcomp/blob/master/imgs/rt_kde.png" alt="model" height=450px>



References: 

[1] Cohen, J. D., Dunbar, K., & McClelland, J. L. (1990). On the control of automatic processes: a parallel distributed processing account of the Stroop effect. Psychological Review, 97(3), 332–361. Retrieved from https://www.ncbi.nlm.nih.gov/pubmed/2200075

[2] the code is based on all stroop model scripts I can possibily find from psyneulink example scripts 
<a href="https://github.com/PrincetonUniversity/PsyNeuLink/tree/master/Scripts">here</a>, 
and most importantly,
<a href="https://github.com/PrincetonUniversity/PsyNeuLink/blob/master/Scripts/Examples/Stroop%20Basic.py">this</a> 
and 
<a href="https://github.com/PrincetonUniversity/PsyNeuLink/blob/master/Scripts/Laura%20Stroop.py">this</a>

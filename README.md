## **AdaBoost**

The binary ADABOOST algorithm is an ensemble (or meta-learning) method that constructs a classifier in an iterative fashion. In each
iteration, it calls a simple learning algorithm (called the base learner) that returns a classifier, and assigns a weight
coefficient to it. The final classification will be decided by a weighted “vote” of the base classifiers. The smaller
the error of the base classifier, the larger is its weight in the final vote. The base classifiers have to be only slightly
better than a random guess (from where their alternative name weak classifier derives), which gives great flexibility
to the design of the base classifier (or feature) set.
For the formal description of ADABOOST, let the training set be Dn = (x1, y1),...,(xn, yn).
The algorithm runs for T iterations. T is the only pre-specified hyper-parameter of ADABOOST that can be set by, for example,
cross-validation. In each iteration t = 1,...,T, we choose a base classifier h
(t)
from a set H of classifiers and
set its coefficient α
(t)
. In the simplest version of the algorithm, H is a finite set of binary classifiers of the form
h : Rd → {−1, 1}, and the base learner executes an exhaustive search over H in each iteration. The output of
ADABOOST is a discriminant function constructed as a weighted vote of the base classifiers - 
f(T )(x) ,T∑t=1α(t)h(t)(x).



Here you can see how the algorithm improves as the steps progress (t=10) <br/>

<br/>
![Figure_5](https://user-images.githubusercontent.com/48287470/105814642-618b3180-5fba-11eb-87d5-d19ae801b8d3.png)

<br/>
<br/>
t=17 <br/>

![Figure_3](https://user-images.githubusercontent.com/48287470/105815267-5684d100-5fbb-11eb-89ec-8b49a2fc9529.png)

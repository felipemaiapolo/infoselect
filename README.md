*InfoSel*: A Python package that makes feature/variable selection for supervised learning tasks using Mutual Information.[¶](#InfoSel:-A-Python-package-that-makes-feature/variable-selection-for-supervised-learning-tasks-using-Mutual-Information.) {#InfoSel:-A-Python-package-that-makes-feature/variable-selection-for-supervised-learning-tasks-using-Mutual-Information.}
======================================================================================================================================================================================================================================================

### *Felipe Maia Polo (felipemaiapolo), Felipe Leno da Silva (f-leno)*[¶](#Felipe-Maia-Polo-(felipemaiapolo),-Felipe-Leno-da-Silva-(f-leno)) {#Felipe-Maia-Polo-(felipemaiapolo),-Felipe-Leno-da-Silva-(f-leno)}

* * * * *

0. Installing *InfoSel* and Loading other Packages[¶](#0.-Installing-InfoSel-and-Loading-other-Packages) {#0.-Installing-InfoSel-and-Loading-other-Packages}
--------------------------------------------------------------------------------------------------------

Installing package:

In [1]:

    !pip install git+https://github.com/felipemaiapolo/infosel.git#egg=infosel

    Requirement already satisfied: infosel from git+https://github.com/felipemaiapolo/infosel.git#egg=infosel in c:\users\felipe\appdata\local\programs\python\python35\lib\site-packages (1.0.0)
    Requirement already satisfied: numpy in c:\users\felipe\appdata\local\programs\python\python35\lib\site-packages (from infosel) (1.18.5)
    Requirement already satisfied: scipy in c:\users\felipe\appdata\local\programs\python\python35\lib\site-packages (from infosel) (1.4.1)
    Requirement already satisfied: matplotlib in c:\users\felipe\appdata\local\programs\python\python35\lib\site-packages (from infosel) (3.0.3)
    Requirement already satisfied: sklearn in c:\users\felipe\appdata\local\programs\python\python35\lib\site-packages (from infosel) (0.0)
    Requirement already satisfied: pandas in c:\users\felipe\appdata\local\programs\python\python35\lib\site-packages (from infosel) (0.24.2)
    Requirement already satisfied: python-dateutil>=2.1 in c:\users\felipe\appdata\local\programs\python\python35\lib\site-packages (from matplotlib->infosel) (2.8.1)
    Requirement already satisfied: cycler>=0.10 in c:\users\felipe\appdata\local\programs\python\python35\lib\site-packages (from matplotlib->infosel) (0.10.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in c:\users\felipe\appdata\local\programs\python\python35\lib\site-packages (from matplotlib->infosel) (1.1.0)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in c:\users\felipe\appdata\local\programs\python\python35\lib\site-packages (from matplotlib->infosel) (2.4.7)
    Requirement already satisfied: scikit-learn in c:\users\felipe\appdata\local\programs\python\python35\lib\site-packages (from sklearn->infosel) (0.22.2.post1)
    Requirement already satisfied: pytz>=2011k in c:\users\felipe\appdata\local\programs\python\python35\lib\site-packages (from pandas->infosel) (2020.1)
    Requirement already satisfied: six>=1.5 in c:\users\felipe\appdata\local\programs\python\python35\lib\site-packages (from python-dateutil>=2.1->matplotlib->infosel) (1.15.0)
    Requirement already satisfied: setuptools in c:\users\felipe\appdata\local\programs\python\python35\lib\site-packages (from kiwisolver>=1.0.1->matplotlib->infosel) (50.3.2)
    Requirement already satisfied: joblib>=0.11 in c:\users\felipe\appdata\local\programs\python\python35\lib\site-packages (from scikit-learn->sklearn->infosel) (0.14.1)

    DEPRECATION: Python 3.5 reached the end of its life on September 13th, 2020. Please upgrade your Python as Python 3.5 is no longer maintained. pip 21.0 will drop support for Python 3.5 in January 2021. pip 21.0 will remove support for this functionality.
    WARNING: You are using pip version 20.2.3; however, version 20.2.4 is available.
    You should consider upgrading via the 'c:\users\felipe\appdata\local\programs\python\python35\python.exe -m pip install --upgrade pip' command.

Loading Packages:

In [2]:

    import infosel as inf
    import numpy as np  
    import pandas as pd
    import matplotlib.pyplot as plt

1. Example of *InfoSel* use[¶](#1.-Example-of-InfoSel-use) {#1.-Example-of-InfoSel-use}
----------------------------------------------------------

**Feature selection** filter techniques help us to decide which features
we should

### 1.1. Dataset[¶](#1.1.-Dataset) {#1.1.-Dataset}

We generate a dataset \$D\$ sampled from
\$\\mathcal{D}=\\{(X\_{0,i},...,X\_{14,i},Y\_i)\\}\_{i=1}\^{n}\$ similar
to the one in
[here](https://www.cs.toronto.edu/~delve/data/add10/desc.html), in which
\$Y\_i\$ is given by

\\begin{align} Y\_i &= f(X\_{0,i},...,X\_{14,i}) + \\epsilon\_i
\\\\[.5em] &=10\\cdot \\sin(\\pi X\_{0,i} X\_{1,i}) + 20
(X\_{2,i}-0.5)\^2 + 10 X\_{3,i} + 5 X\_{4,i} + \\epsilon\_i \\end{align}

Where \$X\_{0,i},...,X\_{14,i} \\overset{iid}{\\sim} U[0,1]\$ and
\$\\epsilon\_i \\sim N(0,1)\$ independent from all the other random
variables for all \$i\\in [n]\$. In the following we set \$n=10000\$:

In [3]:

    def f(X,e): return 10*np.sin(np.pi*X[:,0]*X[:,1]) + 20*(X[:,2]-.5)**2 + 10*X[:,3] + 5*X[:,4] + e

In [5]:

    n=10000

    X = np.random.uniform(0,1,15*n).reshape((n,15))
    e = np.random.normal(0,1,n)
    y = f(X,e)

    X.shape, y.shape

Out[5]:

    ((10000, 15), (10000,))

### 1.2. Selecting Features for a Regression Task[¶](#1.2.-Selecting-Features-for-a-Regression-Task) {#1.2.-Selecting-Features-for-a-Regression-Task}

Training (and validating) GMM:

In [6]:

    %%time

    gmm = inf.get_gmm(X, y, max_comp=10)

    Wall time: 15.2 s

Ordering features by their importances using the *Backward Elimination*
algorithm:

In [7]:

    select = inf.SelectVars(gmm, selection_mode = 'backward')
    select.fit(X, y, verbose=True)    

    Let's start...

    Round =   0   |   Î =  1.39   |   Δ%Î =  0.00   |   Features=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    Round =   1   |   Î =  1.39   |   Δ%Î = -0.00   |   Features=[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    Round =   2   |   Î =  1.39   |   Δ%Î = -0.00   |   Features=[0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14]
    Round =   3   |   Î =  1.39   |   Δ%Î = -0.00   |   Features=[0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14]
    Round =   4   |   Î =  1.39   |   Δ%Î = -0.00   |   Features=[0, 1, 2, 3, 4, 6, 8, 10, 11, 12, 14]
    Round =   5   |   Î =  1.39   |   Δ%Î = -0.00   |   Features=[0, 1, 2, 3, 4, 6, 10, 11, 12, 14]
    Round =   6   |   Î =  1.39   |   Δ%Î = -0.00   |   Features=[0, 1, 2, 3, 4, 10, 11, 12, 14]
    Round =   7   |   Î =  1.38   |   Δ%Î = -0.00   |   Features=[0, 1, 2, 3, 4, 11, 12, 14]
    Round =   8   |   Î =  1.38   |   Δ%Î = -0.00   |   Features=[0, 1, 2, 3, 4, 11, 14]
    Round =   9   |   Î =  1.38   |   Δ%Î = -0.00   |   Features=[0, 1, 2, 3, 4, 11]
    Round =  10   |   Î =  1.38   |   Δ%Î = -0.00   |   Features=[0, 1, 2, 3, 4]
    Round =  11   |   Î =  0.98   |   Δ%Î = -0.29   |   Features=[0, 1, 3, 4]
    Round =  12   |   Î =  0.74   |   Δ%Î = -0.24   |   Features=[0, 1, 3]
    Round =  13   |   Î =  0.39   |   Δ%Î = -0.47   |   Features=[0, 3]
    Round =  14   |   Î =  0.21   |   Δ%Î = -0.47   |   Features=[3]

Checking history:

In [8]:

    select.get_info()

Out[8]:

rounds

mi\_mean

mi\_error

delta

num\_feat

features

0

0

1.392524

0.008824

0.000000

15

[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,...

1

1

1.391535

0.008813

-0.000710

14

[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14]

2

2

1.390415

0.008777

-0.000805

13

[0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14]

3

3

1.389277

0.008775

-0.000819

12

[0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14]

4

4

1.388291

0.008763

-0.000710

11

[0, 1, 2, 3, 4, 6, 8, 10, 11, 12, 14]

5

5

1.387176

0.008752

-0.000803

10

[0, 1, 2, 3, 4, 6, 10, 11, 12, 14]

6

6

1.386056

0.008741

-0.000807

9

[0, 1, 2, 3, 4, 10, 11, 12, 14]

7

7

1.384795

0.008733

-0.000910

8

[0, 1, 2, 3, 4, 11, 12, 14]

8

8

1.383664

0.008720

-0.000817

7

[0, 1, 2, 3, 4, 11, 14]

9

9

1.382456

0.008722

-0.000873

6

[0, 1, 2, 3, 4, 11]

10

10

1.380977

0.008723

-0.001070

5

[0, 1, 2, 3, 4]

11

11

0.982481

0.007915

-0.288561

4

[0, 1, 3, 4]

12

12

0.744129

0.007434

-0.242601

3

[0, 1, 3]

13

13

0.390691

0.007119

-0.474969

2

[0, 3]

14

14

0.206342

0.005372

-0.471853

1

[3]

It is possible to see that the estimated mutual information is untouched
until Round 11, when it varies around -\$30\\%\$.

Since there is a 'break' in Round 10, we should choose to stop the
algorithm at theta round. This will be clear in the Mutual Information
history plot that follows:

In [11]:

    select.plot_mi()

![](data:image/png;base64,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%0A)

Plotting the percentual variations of the mutual information between
rounds:

In [12]:

    select.plot_delta()

![](data:image/png;base64,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%0A)

Making the selection choosing to stop at Round 10:

In [13]:

    X_new = select.transform(X, rd=10)

    X_new.shape

Out[13]:

    (10000, 5)

### 1.3. Selecting Features for a Classification Task[¶](#1.3.-Selecting-Features-for-a-Classification-Task) {#1.3.-Selecting-Features-for-a-Classification-Task}

Categorizing \$Y\$:

In [14]:

    ind0 = (y<np.percentile(y, 33))
    ind1 = (np.percentile(y, 33)<=y) & (y<np.percentile(y, 66))
    ind2 = (np.percentile(y, 66)<=y)

    y[ind0] = 0
    y[ind1] = 1
    y[ind2] = 2

    y=y.astype(int)

In [15]:

    y[:30]

Out[15]:

    array([2, 2, 1, 0, 1, 0, 0, 2, 0, 2, 2, 0, 2, 1, 0, 2, 1, 2, 1, 1, 1, 1,
           0, 2, 0, 1, 1, 1, 2, 1])

Training (and validating) GMMs:

In [16]:

    %%time 

    gmm=inf.get_gmm(X, y, y_cat=True, max_comp=10)

    Wall time: 8.98 s

Ordering features by their importances using the *Backward Elimination*
algorithm:

In [17]:

    select=inf.SelectVars(gmm, selection_mode='forward')
    select.fit(X, y, verbose=True)    

    Let's start...

    Round =   0   |   Î =  0.00   |   Δ%Î =  0.00   |   Features=[]
    Round =   1   |   Î =  0.14   |   Δ%Î =  0.00   |   Features=[3]
    Round =   2   |   Î =  0.28   |   Δ%Î =  1.00   |   Features=[3, 0]
    Round =   3   |   Î =  0.48   |   Δ%Î =  0.75   |   Features=[3, 0, 1]
    Round =   4   |   Î =  0.58   |   Δ%Î =  0.20   |   Features=[3, 0, 1, 4]
    Round =   5   |   Î =  0.65   |   Δ%Î =  0.13   |   Features=[3, 0, 1, 4, 2]
    Round =   6   |   Î =  0.66   |   Δ%Î =  0.00   |   Features=[3, 0, 1, 4, 2, 8]
    Round =   7   |   Î =  0.66   |   Δ%Î =  0.00   |   Features=[3, 0, 1, 4, 2, 8, 13]
    Round =   8   |   Î =  0.66   |   Δ%Î =  0.00   |   Features=[3, 0, 1, 4, 2, 8, 13, 7]
    Round =   9   |   Î =  0.66   |   Δ%Î =  0.00   |   Features=[3, 0, 1, 4, 2, 8, 13, 7, 10]
    Round =  10   |   Î =  0.66   |   Δ%Î =  0.00   |   Features=[3, 0, 1, 4, 2, 8, 13, 7, 10, 9]
    Round =  11   |   Î =  0.66   |   Δ%Î =  0.00   |   Features=[3, 0, 1, 4, 2, 8, 13, 7, 10, 9, 11]
    Round =  12   |   Î =  0.67   |   Δ%Î =  0.00   |   Features=[3, 0, 1, 4, 2, 8, 13, 7, 10, 9, 11, 5]
    Round =  13   |   Î =  0.67   |   Δ%Î =  0.00   |   Features=[3, 0, 1, 4, 2, 8, 13, 7, 10, 9, 11, 5, 12]
    Round =  14   |   Î =  0.67   |   Δ%Î =  0.00   |   Features=[3, 0, 1, 4, 2, 8, 13, 7, 10, 9, 11, 5, 12, 6]
    Round =  15   |   Î =  0.67   |   Δ%Î = -0.00   |   Features=[3, 0, 1, 4, 2, 8, 13, 7, 10, 9, 11, 5, 12, 6, 14]

Checking history:

In [18]:

    select.get_info()

Out[18]:

rounds

mi\_mean

mi\_error

delta

num\_feat

features

0

0

0.000000

0.000000

0.000000

0

[]

1

1

0.138134

0.005067

0.000000

1

[3]

2

2

0.276204

0.006440

0.999527

2

[3, 0]

3

3

0.484021

0.006614

0.752406

3

[3, 0, 1]

4

4

0.581658

0.006284

0.201721

4

[3, 0, 1, 4]

5

5

0.654569

0.005383

0.125350

5

[3, 0, 1, 4, 2]

6

6

0.656105

0.005392

0.002346

6

[3, 0, 1, 4, 2, 8]

7

7

0.657538

0.005410

0.002184

7

[3, 0, 1, 4, 2, 8, 13]

8

8

0.658846

0.005409

0.001989

8

[3, 0, 1, 4, 2, 8, 13, 7]

9

9

0.660517

0.005411

0.002536

9

[3, 0, 1, 4, 2, 8, 13, 7, 10]

10

10

0.662341

0.005418

0.002761

10

[3, 0, 1, 4, 2, 8, 13, 7, 10, 9]

11

11

0.664055

0.005423

0.002588

11

[3, 0, 1, 4, 2, 8, 13, 7, 10, 9, 11]

12

12

0.665984

0.005526

0.002904

12

[3, 0, 1, 4, 2, 8, 13, 7, 10, 9, 11, 5]

13

13

0.667485

0.005605

0.002255

13

[3, 0, 1, 4, 2, 8, 13, 7, 10, 9, 11, 5, 12]

14

14

0.669384

0.005665

0.002845

14

[3, 0, 1, 4, 2, 8, 13, 7, 10, 9, 11, 5, 12, 6]

15

15

0.668237

0.005923

-0.001714

15

[3, 0, 1, 4, 2, 8, 13, 7, 10, 9, 11, 5, 12, 6,...

It is possible to see that the estimated mutual information is untouched
from Round 6 onwards.

Since there is a 'break' in Round 5, we should choose to stop the
algorithm at theta round. This will be clear in the Mutual Information
history plot that follows:

In [19]:

    select.plot_mi()

![](data:image/png;base64,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%0AREQk5f8DTwFpPWvsq6gAAAAASUVORK5CYII=%0A)

Plotting the percentual variations of the mutual information between
rounds:

In [20]:

    select.plot_delta()

![](data:image/png;base64,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%0A)

Making the selection choosing to stop at Round 5:

In [21]:

    X_new = select.transform(X, rd=5)

    X_new.shape

Out[21]:

    (10000, 5)

* * * * *

2. Mathematical Details[¶](#2.-Mathematical-Details) {#2.-Mathematical-Details}
----------------------------------------------------

### 2.1. Justification on using Mutual Information for feature selection[¶](#2.1.-Justification-on-using-Mutual-Information-for-feature-selection) {#2.1.-Justification-on-using-Mutual-Information-for-feature-selection}

Recall that the Mutual Information between \$Y\$ and a random vector
\$\\mathbf{X}'\$, sampled from \$p\_{\\mathbf{X}',Y}\$, is given by

\\begin{align} I(Y;\\mathbf{X}')&=\\underset{(\\mathbf{X}',Y)\\sim
p\_{\\mathbf{X}',Y}}{\\mathbb{E}} \\left[\\text{log
}\\frac{p\_{\\mathbf{X}',Y}(\\mathbf{X}',Y)}{p\_{\\mathbf{X}'}(\\mathbf{X}')p\_{Y}(Y)}
\\right] \\end{align}

The following Theorem shows the theoretical foundation of using Mutual
Information between labels and features as a guide in a feature
selection procedure:

**Theorem 1**: Consider a random vector
\$\\mathbf{X}=(\\mathbf{X}\_1,\\mathbf{X}\_2)\$ and a random variable
\$Y\$ with joint p.d.f. \$p\_{\\mathbf{X},Y}\$. Then
\$I(Y;\\mathbf{X})\\geq I(Y;\\mathbf{X}\_1)\$ and \$I(Y;\\mathbf{X})=
I(Y;\\mathbf{X}\_1)\$ if and only if \$Y\$ is conditionally independent
of \$\\mathbf{X}\$ given \$\\mathbf{X}\_1\$.

The proof can be found in [1]. Theorem 1 tell us that if we find a
subset of features \$\\mathbf{X}\_1\$ from original set
\$\\mathbf{X}=(\\mathbf{X}\_1,\\mathbf{X}\_2)\$ that \$I(Y;\\mathbf{X})=
I(Y;\\mathbf{X}\_1)\$ holds, then working with this subset can satisfy
our needs, as this set is statistically sufficient for the target
variable \$Y\$. In practice, in order to select a good subset of
features \$\\mathbf{X}\_1\$, it can be the case that we are happy with
\$I(Y;\\mathbf{X})\> I(Y;\\mathbf{X}\_1)\$, and that will depend on how
much information we are willing to retrieve in exchange for a greater
number of features.

### 2.2. Details on using Gaussian Mixture Models (GMMs)[¶](#2.2.-Details-on-using-Gaussian-Mixture-Models-(GMMs)) {#2.2.-Details-on-using-Gaussian-Mixture-Models-(GMMs)}

Consider \$Y\$ continuous. We model the distributions as GMMs, i.e.

\\begin{align\*} \\\\ p\_\\theta(\\mathbf{x},y)=\\sum\_{k=1}\^K
\\alpha\^{(k)} \\cdot N\\Bigg(\\begin{bmatrix} \\mathbf{x}\\\\ y
\\end{bmatrix}\~\\Big | \~
\\boldsymbol{\\mu}\^{(k)}\_{\\mathbf{X},Y},\\boldsymbol{\\Sigma}\^{(k)}\_{\\mathbf{X},Y}
\\Bigg) \\\\ \\\\ \\end{align\*}

Where \$\\sum\_{k=1}\^K \\alpha\^{(k)} =1\$ and \$\\alpha\^{(k)}\\geq 0,
\\forall k \\in [K]\$. The parameters \$\\alpha\^{(k)},\~
\\boldsymbol{\\mu}\^{(k)}\_{\\mathbf{X},Y}\$ and
\$\\boldsymbol{\\Sigma}\^{(k)}\_{\\mathbf{X},Y}\$ are learned via EM
algorithm (Scikit-Learn Implementation). This method is very efficient
since we need to fit the GMM only once. In this case, we fit the model
considering the whole set of features. Right after fitting the model, we
choose a small subset of features with 'enough' information according to
the reward function.

Now consider the case in which \$Y \\in \\{0,1,...,C-1 \\}\$ . We model
the distributions in the following way:

\\begin{align\*} \\\\ p\_\\theta(\\mathbf{x},y)&=
\\mathbb(Y=y)p\_\\theta(\\mathbf{x}|Y=y)\\\\ &= \\mathbb(Y=y)
\\sum\_{k=1}\^{K\_y} \\alpha\^{(k,y)} \\cdot N\\Big(\\mathbf{x} \~\\big
| \~
\\boldsymbol{\\mu}\^{(k,y)}\_{\\mathbf{X}},\\boldsymbol{\\Sigma}\^{(k,y)}\_{\\mathbf{X}}
\\Big) \\\\ \\\\ \\end{align\*}\\begin{align\*} \\\\
p\_\\theta(\\mathbf{x})&=\\sum\_{y=0}\^{C-1}
\\mathbb{P}(Y=y)p\_\\theta(\\mathbf{x}|Y=y)\\\\ &= \\sum\_{y=0}\^{C-1}
\\mathbb{P}(Y=y) \\sum\_{k=1}\^{K\_y} \\alpha\^{(k,y)} \\cdot
N\\Big(\\mathbf{x} \~\\big | \~
\\boldsymbol{\\mu}\^{(k,y)}\_{\\mathbf{X}},\\boldsymbol{\\Sigma}\^{(k,y)}\_{\\mathbf{X}}
\\Big) \\\\ \\\\ \\end{align\*}

In this case, we need to fit \$C\$ GMMs, one for each value of \$Y\$;

* * * * *

3. References[¶](#3.-References) {#3.-References}
--------------------------------

[1]

[2]

[3]

Escolhendo número de componentes do GMM

O número ótimo de componentes é **k\_star**. Fitando GMM final com
k\_star componentes

Exemplo 2: Classificação[¶](#Exemplo-2:-Classificação) {#Exemplo-2:-Classificação}
------------------------------------------------------

Abrindo dataset

In [13]:

    X, y = data['fried_delve']

    X.shape, y.shape

Out[13]:

    ((15000, 10), (15000,))

In [17]:

    select.plot_mi()
    select.plot_delta()

![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAYUAAAEKCAYAAAD9xUlFAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz%0AAAALEgAACxIB0t1+/AAAIABJREFUeJzt3Xl8FeXZ//HPRdj3LaKyL0FFUJGAaK21FS0uldYVrAti%0Ai/URpXWp2j6PrdrnZ61aW1tcqFLQihRRKyqPuGGtO2GV3YgIQZGwyhqyXL8/ziQeY0gGyGSSc77v%0A1yuvnJm5Z873KHCdmXvmvs3dERERAagXdwAREak9VBRERKSMioKIiJRRURARkTIqCiIiUkZFQURE%0AykRaFMxsqJktN7NcM7u5gu1dzGyWmc0zs4VmdkaUeUREpHIW1XMKZpYBrABOBfKA2cAId1+S1GY8%0AMM/dHzSzPsAMd+8WSSAREalSlGcKg4Bcd1/p7nuAKcCwcm0caBm8bgV8FmEeERGpQv0Ij90RWJO0%0AnAccV67Nb4GXzewaoBkwpKqDtm/f3rt161ZNEUVE0sOcOXM2uHtmVe2iLAphjAAmuvu9ZnY88LiZ%0A9XX3kuRGZjYaGA3QpUsXcnJyYogqIlJ3mdmnYdpFefloLdA5ablTsC7ZFcBUAHd/F2gMtC9/IHcf%0A7+7Z7p6dmVlloRMRkf0UZVGYDWSZWXczawgMB6aXa7MaOAXAzI4gURTyI8wkIiKViKwouHsRMAaY%0ACSwFprr7YjO73czODppdD/zUzBYATwIjXcO2iojEJtI+BXefAcwot+7WpNdLgG9FmUFERMLTE80i%0AIlJGRUFERMqoKIiISBkVBRERKRP3w2siKWPD9gJ2FhQD4Dil99E54O6U3laXWF9+O1/br/wySfuW%0AHinxGn79zIc48Lsf9f3asUuPW/reZfsG25KP91X74Oj+9Syl25Lz3vvycgB+cWrvCnOVtS+/rYL3%0A+tpnL5errEnS55jw9icAjPpWd8zAAMyAxOvgJYZ9tT1Yb5RtDNp+fb/y+1K2r3H/qysAuO60w6iX%0AeFPqGdQzo169RJt6lniHepbYZsHvevWs7P3K9jEre8/S5dJ9vlqX+H3N5LnUz6jHtKtOIEqRDYgX%0AlezsbNcTzVLbnHz3LFZt3Bl3DElx3do15Y0bv7tf+5rZHHfPrqqdzhREDtCbK/JZtXEnLRvX59Yf%0AHAmE+9ZZ/tvrV21L97Ok7V/fr6ytwV3/twyAm884ouybqJU7bunhkpfLviHb198zeXv543yVH26a%0AthAzuPv8oyv8Rl7RsSr8vFbBt/VyuZP/GyZnTz4Lgq/OhBKvKz6zSj6DKVXRmUn5MyNwrvvnAhzn%0AnvOPocSdEv/qbKp0uSTp7Kyk5KvlkqCNE/x2p6Tkq3XJbcqOkbT8wKxcmjeK/p9snSmIHICPvtjG%0AOQ+8Q8c2TZh21Qk18pdWZH+EPVNQR7PIftq4vYBRk2bTqEEGj44cqIIgKUFFQWQ/FBQVc+Xjc1j/%0AZQF/u3QAHVs3iTuSSLXQVxuRfeTu3Pz0h+R8upm/XtSf/l3axB1JpNroTEFkH/319VyenbeW60/t%0AzVlHHRp3HJFqpaIgsg9eWPgZ976ygnP6d2TM93rFHUek2qkoiIQ0b/Vmrp+6gIHd2nDnuf3KbosU%0ASSUqCiIh5G3eyU8fy6FDy8Y8fEk2jepnxB1JJBLqaBapwrbdhVwxMYeCohKmjM6mbbOGcUcSiYyK%0AgkgliopLuObJeeTmb2fS5YPodVCLuCOJRCrSy0dmNtTMlptZrpndXMH2+8xsfvCzwsy2RJlHZF/9%0A7sWlvLE8n9uHHcmJWe3jjiMSucjOFMwsAxgHnArkAbPNbHowBScA7v6LpPbXAP2jyiOyrx5/dxUT%0A31nFFSd258fHdY07jkiNiPJMYRCQ6+4r3X0PMAUYVkn7EcCTEeYRCe3fK/L57fNLGHLEQfzqjCPi%0AjiNSY6IsCh2BNUnLecG6bzCzrkB34PW9bB9tZjlmlpOfn1/tQUWSrfhiG2OemEvvDi348/D+ZNTT%0AraeSPmrLLanDgWnuXlzRRncf7+7Z7p6dmZlZw9EknWzYXsCoibNp3DCDRy/LppkGuZM0E2VRWAt0%0ATlruFKyryHB06UhitruwmNGP5bBhewGPXJrNoRrkTtJQlEVhNpBlZt3NrCGJf/inl29kZocDbYB3%0AI8wiUil355fTFjJ39Rb+eMExHN25ddyRRGIRWVFw9yJgDDATWApMdffFZna7mZ2d1HQ4MMXr2mw/%0AklL+/NpHTF/wGTd+/zDO6HdI3HFEYhPpBVN3nwHMKLfu1nLLv40yg0hVnpu/lj+9+hHnHtuJ/zq5%0AZ9xxRGJVWzqaRWIx59PN3DhtIYO6teX/ndNXg9xJ2lNRkLS1ZtNORj+WwyGtGvPQJQM0yJ0IKgqS%0Apr7cXcgVk2ZTWFzChJEDNcidSEA3YUvaKSouYczkeazM38FjowbRM7N53JFEag0VBUk7d7ywhDdX%0A5PP7c/pxQi8NcieSTJePJK1MemcVk979lNEn9WD4oC5xxxGpdVQUJG3MWr6e255fzJAjOnDT0MPj%0AjiNSK6koSFpYvm4b10yex+EHt+TPw4/RIHcie6GiICkvf1tikLtmjTJ4dKQGuROpjP52SErbXVjM%0A6Mdz2LRjD1OvPJ5DWmmQO5HKqChIynJ3bpy2kHmrt/DQxQPo16lV3JFEaj1dPpKUdd+rH/H8gs+4%0AaejhDO17cNxxROoEFQVJSf+at5b7X/uIC7I78bPv9Ig7jkidoaIgKSdn1SZ+OW0hg3u05Xc/7KdB%0A7kT2gfoUJKUM++tbLP7sSzq3bcpDFw+gYX197xHZF/obIynjkw07WLZuGw48elk2rZtqkDuRfaWi%0AIClhYd4WznvwHYpLnMMPbkEPDXInsl8iLQpmNtTMlptZrpndvJc2F5jZEjNbbGaTo8wjqek/H+Uz%0AfPx7NG6Qwcu/OIkXr/123JFE6qzI+hTMLAMYB5wK5AGzzWy6uy9JapMF3AJ8y903m9lBUeWR1PTc%0A/LXc8NQCemY2Z9KoQXRo2TjuSCJ1WpRnCoOAXHdf6e57gCnAsHJtfgqMc/fNAO6+PsI8kmImvPUJ%0AY6fMp3+XNvzzyuNVEESqQZRFoSOwJmk5L1iXrDfQ28zeNrP3zGxoRQcys9FmlmNmOfn5+RHFlbrC%0A3bnrpWXc/sISvn9kBx4bNYhWTRrEHUskJcR9S2p9IAs4GegEvGlm/dx9S3Ijdx8PjAfIzs72mg4p%0AtUdhcQm3PPMh0+bkcdFxXbhjWF+NeCpSjaIsCmuBzknLnYJ1yfKA9929EPjEzFaQKBKzI8wlddSu%0APcVcPXkury9bz8+HZDH2lCw9mCZSzaK8fDQbyDKz7mbWEBgOTC/X5l8kzhIws/YkLietjDCT1FGb%0Ad+zhokfe443l6/ndD/vy8yG9VRBEIhDZmYK7F5nZGGAmkAFMcPfFZnY7kOPu04Ntp5nZEqAYuNHd%0AN0aVSeqmtVt2cemj77Nm8y4e+PGxDO17SNyRRFKWudetS/TZ2dmek5MTdwypISu+2Malj37AjoIi%0A/nZZNoN7tIs7kkidZGZz3D27qnZxdzSL7FXOqk2MmjibRg0ymPqz4znikJZxRxJJeSoKUiu9suQL%0Axkyey6Gtm/DYqEF0bts07kgiaUFFQWqdf85ezS3PfEi/jq2YMHIg7Zo3ijuSSNpQUZBaw90ZNyuX%0Ae15ewUm9M3nwx8fSrJH+iIrUJP2Nk1qhpMS57fnFTHr3U37UvyN3nXuU5kIQiUGoomBmJwDdktu7%0A+2MRZZI0U1BUzHVTF/Diws/56be7c8vpR1BPTymLxKLKomBmjwM9gfkkniUAcEBFQQ7Ytt2FXPn4%0AHN75eCO/OuNwRp/UM+5IImktzJlCNtDH69oDDVLrrd+2m8v/Pptl67Zx7/lHc+6ATnFHEkl7YYrC%0AIuBg4POIs0gaWbVhB5dO+ID8bQU8clk23z1MU2mI1AZhikJ7YImZfQAUlK5097MjSyUpbdHarYz8%0A+wcUlziTf3oc/bu0iTuSiATCFIXfRh1C0sdbH23gysdzaN20IY9dMYiemktZpFapsii4+7/NrAMw%0AMFj1gWZIk/3x/ILPuG7qfHq0T0ydeXArzZQmUttUeSO4mV0AfACcD1wAvG9m50UdTFLLxLc/4dop%0A8+jfuQ1Tf3a8CoJILRXm8tGvgYGlZwdmlgm8CkyLMpikBnfn7pnLeeCNjzmtTwfuH9Gfxg0y4o4l%0AInsRpijUK3e5aCPRTs4jKaKouIRfPfshU3PyGDGoC3cMO5L6GfqjI1KbhSkKL5nZTODJYPlCYEZ0%0AkSQV7NpTzJjJc3lt2XquPSWLXwzR1JkidUGYjuYbzexc4FvBqvHu/my0saQu211YzMD/fZXtBUXc%0A8cO+XDK4a9yRRCSkUOfy7v60u18X/IQuCGY21MyWm1mumd1cwfaRZpZvZvODn5/sS3ipnR57dxXb%0AC4romdlMBUGkjtnrmYKZveXuJ5rZNhJjHZVtAtzdK50Gy8wygHHAqUAeMNvMprv7knJN/+nuY/Yv%0AvtQ2X+4u5IE3PqZVkwa01zwIInXOXouCu58Y/G6xn8ceBOS6+0oAM5sCDAPKFwVJIY+8uZItOwt5%0A4ZoT6duxVdxxRGQfhXlO4fEw6yrQEViTtJwXrCvvXDNbaGbTzKzzXjKMNrMcM8vJz88P8dYSh/xt%0ABTzy1iecedQhKggidVSYPoUjkxfMrD4woJre/3mgm7sfBbwCTKqokbuPd/dsd8/OzMyspreW6jZu%0AVi4FRSVcf2rvuKOIyH7aa1Ews1uC/oSjzOzL4Gcb8AXwXIhjrwWSv/l3CtaVcfeN7l46yN4jVF+x%0AkRq2ZtNOnnj/U84f0IkeGs9IpM7aa1Fw9zuD/oS73b1l8NPC3du5+y0hjj0byDKz7mbWEBgOTE9u%0AYGaHJC2eDSzdj88gtcCfXv0IM2PskKy4o4jIAQjznMItZtYGyAIaJ61/s4r9isxsDDATyAAmuPti%0AM7sdyHH36cC1ZnY2UARsAkbu9yeR2Kz4YhvPzsvjihO7c0irJnHHEZEDEGY6zp8AY0lc/pkPDAbe%0ABb5X1b7uPoNyTz+7+61Jr28Bwpx1SC12z8zlNG1Yn6tO7hV3FBE5QGE6mseSGDb7U3f/LtAf0C1A%0AAsC81Zt5eckXjD6pB22bNYw7jogcoDBFYbe77wYws0buvgw4LNpYUhe4O394aTntmjVk1Ind444j%0AItUgzIB4eWbWGvgX8IqZbQY+izaW1AVv5W7g3ZUb+c0P+tC8UZg/SiJS24XpaP5R8PK3ZjYLaAW8%0AFGkqqfVK50no2LoJFx3XJe44IlJNQg2IZ2ZtzOwoYBuJJ5P7RppKar2XFq1jYd5Wfj4ki0b1NWmO%0ASKoIc/fRHSRuFV0JlASrnRB3H0lqKiou4e6Xl5N1UHPOObZT3HFEpBqFuRB8AdDT3fdEHUbqhmfm%0ArmVl/g4eungAGfU0cY5IKglz+WgR0DrqIFI37C4s5r5XV3B059Z8/8gOcccRkWoW5kzhTmCemS0C%0ASscpwt3PjiyV1Fr/eO9TPt+6m3vPP1rTa4qkoDBFYRJwF/AhX/UpSBraFkygc2Kv9pzQq33ccUQk%0AAmGKwgZ3vz/yJFLrPfKfT9i0Yw83fl/PLoqkqjBFYY6Z3UlihNPky0dzI0sltc7G7QU88p+VnN73%0AYI7urC4mkVQVpij0D34PTlqnW1LTzANvfMyuwmKuP00T6IikskqLgpnVAx5096k1lEdqobVbdvH4%0Au59y3oBO9Dpof6fsFpG6oNJbUt29BBhTQ1mklvrzqysAGDtEZwkiqS7McwqvmNkNZtbZzNqW/kSe%0ATGqF3PXbmDYnj4sHd6Vja02gI5LqwvQpjAp+X520zoEe1R9Hapt7X15BkwYZXP3dnnFHEZEaUOWZ%0Agrt3r+AnVEEws6FmttzMcs3s5kranWtmbmbZ+xJeorVgzRb+b9E6fvLtHrRr3ijuOCJSA8IMiNcA%0AuAo4KVj1BvCwuxdWsV8GMA44lcTIqrPNbLq7LynXrgWJ2d3e3+f0Eqm7Zy6nTdMG/OTbmkBHJF2E%0A6VN4EBgAPBD8DAjWVWUQkOvuK4PB9KYAwypodweJJ6Z3h0osNeKd3A28lbuBq7/bixaNG8QdR0Rq%0ASJg+hYHufnTS8utmtiDEfh2BNUnLecBxyQ3M7Figs7u/aGY37u1AZjYaGA3QpYsmdImau3PXzOUc%0A2qoxFw/uGnccEalBYc4Uis2srJfRzHoAxQf6xsEzEH8Erq+qrbuPd/dsd8/OzMw80LeWKsxc/AUL%0A1mxh7JAsGjfQBDoi6STMmcKNwCwzWwkY0BW4PMR+a4HOScudgnWlWpCYwe2NYLTNg4HpZna2u+eE%0AOL5EoLjEuffl5fTIbMa5mkBHJO3stSiY2fnu/hSJGdeygNJR0Ja7e8He9ksyG8gys+4kisFw4KLS%0Aje6+FSgbatPM3gBuUEGI17Pz1vLR+u088ONjqZ8RarZWEUkhlf2tvyX4/bS7F7j7wuAnTEHA3YtI%0APA09E1gKTHX3xWZ2u5lpLoZaqKComPteWUG/jq04ve/BcccRkRhUdvloo5nNArqb2fTyG8NMsuPu%0AM4AZ5dbdupe2J1d1PInW5PdXs3bLLn5/bj9NoCOSpiorCmcCxwKPA/fWTByJy/aCIv76ei7H92jH%0AiZpARyRt7bUoBM8WvGdmJ7h7fg1mkhhMeOsTNu7Ywy+HHqazBJE0FubuozZm9r9At+T27q75FFLE%0Aph17+NubKzmtTwf6d2kTdxwRiVGYovAU8BDwCNXwfILUPg++kcv2PUXcoGk2RdJemKJQ5O5hhrWQ%0AOujzrbuY9O6nnNO/E707aAIdkXQX5kb0583sv8zsEM2nkHruf+0j3J2fD8mKO4qI1AJhzhQuC34n%0Aj02k+RRSwMr87UzNyeOSwV3p3LZp3HFEpBaosii4u8ZNTlH3vrKCRvXrcfV3e8UdRURqicqGuTin%0Ash3d/ZnqjyM1ZdHarby48HOu+V4vMltoAh0RSajsTOEHlWxzQEWhDvvDzOW0btqAn56kq4Ai8pXK%0AHl4LMxKq1EHvfryRN1fk86szDqelJtARkSQaBjPNuDt/mLmMg1s25tLju8UdR0RqGRWFNPPq0vXM%0AW60JdESkYioKaaS4xLln5nK6t2/G+QM0gY6IfJPuPkojz81fy/IvtvGXEf01gY6IVEh3H6WJPUUl%0A3PfqCo48tCVn9jsk7jgiUkvp7qM0MWX2atZs2sXEy/tSr56GxhaRioUZ5gIzOxM4Emhcus7dbw+x%0A31Dgz0AG8Ii7/77c9p8BV5MYfXU7MNrdl4ROL6Hs3FPE/a/lMqh7W77TOzPuOCJSi1V5YdnMHgIu%0ABK4BDDgf6BpivwxgHHA60AcYYWZ9yjWb7O793P0Y4A/AH/ctvoTx97dXsWF7ATdpAh0RqUKY3sYT%0A3P1SYLO73wYcD3QOsd8gINfdVwazuE0BhiU3cPcvkxabkeirkGq0ZeceHvr3xww54iAGdNXgtiJS%0AuTCXj3YFv3ea2aHARiDMIHkdgTVJy3nAceUbmdnVwHVAQ6DC2dzMbDQwGqBLly4h3lpKPfjvj9le%0AoAl0RCScMGcKL5hZa+BuYC6wisS3/mrh7uPcvSdwE/Dfe2kz3t2z3T07M1PXxMP60bi3Gf/mSn54%0ATEcOP7hl3HFEpA4IM3T2HcHLp83sBaCxu28Ncey1fP0yU6dg3d5MATTDWzUpLnFWbdwBDr8Y0jvu%0AOCJSR1RZFMzs0grW4e6PVbHrbCDLzLqTKAbDgYvKHSfL3T8KFs8EPkIOmLtzxwtL2LyzkC5tm9Kl%0AnSbQEZFwwvQpDEx63Rg4hcRlpEqLgrsXmdkYYCaJW1InuPtiM7sdyHH36cAYMxsCFAKb+WqWNzkA%0Af/vPSia+s4orTuzO/5xV/oYvEZG9C3P56Jrk5aB/YVKYg7v7DGBGuXW3Jr0eGy6mhPXc/LX8vxnL%0AOPOoQ/j1GUfEHUdE6pj9GQBnB6CL1LXQOx9v4IanFjCoe1vuPf9oPbksIvssTJ/C83z1/EA9Eg+i%0APRVlKNl3y9Z9yZWPzaFbu2b87ZJsDYstIvslTJ/CPUmvi4BP3T0vojyyHz7fuouRE2bTtFEGE0cN%0AolVTzaYmIvsnzOWjM9z938HP2+6eZ2Z3RZ5MQtm6q5CRE2azvaCIv48cRMfWTeKOJCJ1WJiicGoF%0A606v7iCy7wqKirny8RxWbtjOw5cMoM+hekBNRA5MZZPsXAX8F9DTzBYmbWoBvB11MKlcSYlz41ML%0AeW/lJv504TF8q1f7uCOJSAqorE9hMvB/wJ3AzUnrt7n7pkhTSZXuemkZ0xd8xi+HHsYP+3eMO46I%0ApIjKJtnZCmw1s5vKbWpuZs3dfXW00WRvJr79CQ+/uZJLBnflqu/0jDuOiKSQMHcfvUjillQj8URz%0Ad2A5iUl3pIa9tOhzbnthCaf26cBvzz5S8yOISLUK80Rzv+RlMzsWuDKyRLJXOas2MXbKfI7p3Jr7%0Ah/cnQw+niUg12+cnmt19LpAdQRapRO767VwxKYdDWzfh0csG0qShHk4TkeoX5onm65IW6wHHAhsi%0ASyTfsP7L3Vw24QMaZBiTLh9E22YN444kIikqTJ9Ci6TXRST6GJ6OJo6Ut72giMsnzmbzzj1MGT1Y%0Aw2CLSKTC9CncVhNB5JsKi0v4ryfmsmzdNh65LJujOrWOO5KIpLjKHl6bXtmO7n529ceRUu7OzU9/%0AyJsr8vnDuUfx3cMOijuSiKSBys4UjgfWAE8C75O4JVVqyB9fWcHTc/P4+ZAsLhjYueodRESqQWVF%0A4WAS4x6NIDGN5ovAk+6+uCaCpbPJ76/mL6/ncmF2Z8aekhV3HBFJI3u9JdXdi939JXe/DBgM5AJv%0ABFNshmJmQ81suZnlmtnNFWy/zsyWmNlCM3vNzLru16dIIa8t/YL//teHnHxYJr/7UV89nCYiNarS%0A5xTMrJGZnQP8A7gauB94NsyBzSwDGEdiRNU+wAgzKz9h8Dwg292PAqYBf9i3+Kll/potjJk8jyMP%0AbcW4i46lQcb+TIwnIrL/KutofgzoS2KO5dvcfdE+HnsQkOvuK4PjTQGGAUtKG7j7rKT27wEX7+N7%0ApIxVG3YwauJsMls0YsLIgTRrFOZuYRGR6lXZV9GLgSxgLPCOmX0Z/Gwzsy9DHLsjiY7qUnnBur25%0AgsSorN9gZqPNLMfMcvLz80O8dd2ycXsBl/39A9ydiZcPJLNFo7gjiUiaqmyU1Bq7dmFmF5MYOuM7%0Ae8kyHhgPkJ2d7RW1qat27ili1KQc1m3dzeSfDqZHZvO4I4lIGovyGsVaIPleyk7Buq8xsyHAr4Hv%0AuHtBhHlqnaLiEq6ZPI8P87bw0MUDGNC1TdyRRCTNRXk2MBvIMrPuZtYQGA587YE4M+sPPAyc7e7r%0AI8xS67g7//PcYl5btp7bhvXltCMPjjuSiEh0RcHdi4AxwExgKTDV3Reb2e1mVvo09N1Ac+ApM5tf%0A1VPUqWTcrFye/GA1V53ck0sGp/2duCJSS0R6i4u7zyBx91LyuluTXg+J8v1rq2lz8rjn5RX8qH9H%0Afvn9w+KOIyJSRjfC17A3V+Rz89MLObFXe+469yg9nCYitYqKQg1atHYrV/1jDr0Oas6DFx9Lw/r6%0Azy8itYv+Vaohazbt5PKJs2nVpAGTRg2iReMGcUcSEfkGFYUasGXnHkb+/QMKCouZNGoQHVo2jjuS%0AiEiFNJZCxPYUlfDtu2axvaCIKaMHk9WhRdU7iYjERGcKEfvDS8vYVlBEz8xmHNejXdxxREQqpaIQ%0AoVeXfMEjb33Cpcd35dXrT447johIlVQUIvLZll3cMG0BfQ5pya/OOCLuOCIioagoRKCwuIRrnpxH%0AYVEJ4358LI0bZMQdSUQkFHU0R+CPr6xgzqebuX9Ef7q3bxZ3HBGR0HSmUM3eWL6eB9/4mBGDOnP2%0A0YfGHUdEZJ+oKFSjdVt3c93UBRx+cAt+84Mj444jIrLPVBSqSVFxCddOmceuPcX89SL1I4hI3aQ+%0AhWpy/2sf8cEnm7j3/KPpdZBmTxORuklnCtXg7dwN/GVWLucN6MS5AzrFHUdEZL+pKByg9dt2M3bK%0AfHpmNuf2YepHEJG6TZePDkBxifOLf85ne0EhT/zkOJo21H9OEanbIj1TMLOhZrbczHLN7OYKtp9k%0AZnPNrMjMzosySxQemJXL27kbue3sIznsYA10JyJ1X2RFwcwygHHA6UAfYISZ9SnXbDUwEpgcVY6o%0AvLdyI/e9uoIfHnMoF2R3jjuOiEi1iPJ6xyAg191XApjZFGAYsKS0gbuvCraVRJij2m3cXsDYKfPo%0A1q4Zv/tRP02pKSIpI8rLRx2BNUnLecG6fWZmo80sx8xy8vPzqyXc/iopca6buoDNOwv5y0X9ad5I%0A/QgikjrqxN1H7j7e3bPdPTszMzPWLA+/uZJ/r8jnf87qw5GHtoo1i4hIdYuyKKwFki+2dwrW1Vk5%0AqzZxz8vLObPfIVx8XJe444iIVLsoi8JsIMvMuptZQ2A4MD3C94vU5h17uPbJeXRs3YQ7z1U/goik%0ApsiKgrsXAWOAmcBSYKq7Lzaz283sbAAzG2hmecD5wMNmtjiqPAfC3blx2gLytxfw14v607Jxg7gj%0AiYhEItJeUnefAcwot+7WpNezSVxWqtUefesTXl26nt/8oA9HdWoddxwRkcjUiY7mOM1fs4W7XlrG%0AaX06MPKEbnHHERGJlIpCJbbuKmTM5Lkc1KIxd593tPoRRCTl6Sb7vXB3bpq2kHVbd/PUz46nVVP1%0AI4hI6tOZwl489u6nvLR4HTcNPZz+XdrEHUdEpEaoKFRg0dqt/O+LS/ne4QdxxYnd444jIlJjVBTK%0A2ba7kKsnz6Vd84bce/7R1KunfgQRSR/qU0ji7tzyzIfkbd7FlNGDadOsYdyRRERqlM4Ukkz+YDUv%0ALPyc607tzcBubeOOIyJS41QUAks//5Lbnl/Ct7Pac9V3esYdR0QkFioKwI6CIq6ePJfWTRpw34XH%0AqB9BRNJW2vcpuDv//a9FrNqwgyd+Mpj2zRvFHUlEJDZpf6bw1Jw8np23lrGn9Ob4nu3ijiMiEqu0%0ALgorvtiMdF+5AAAHFElEQVTGrc8t4oSe7RjzvV5xxxERiV3aFoVde4q5+om5NG9Unz8NP4YM9SOI%0AiKRvn8Jvpi8iN387j486joNaNI47johIrZCWZwrPzstjak4eV5/cixOz2scdR0Sk1ki7ovBx/nZ+%0A/ewiBnVry8+HZMUdR0SkVom0KJjZUDNbbma5ZnZzBdsbmdk/g+3vm1m3KPPsLkz0IzSqX4/7R/Sn%0Afkba1UQRkUpF9q+imWUA44DTgT7ACDPrU67ZFcBmd+8F3AfcFVUegDteWMKyddv444XHcHAr9SOI%0AiJRn7h7Ngc2OB37r7t8Plm8BcPc7k9rMDNq8a2b1gXVAplcSqkX3Fj7gNwO+tu6s3mdxwwk3AHDy%0AxJO/sc9Zvc/i8OYXMWbyPFq2fZ+2B725z/tru7Zru7bX5e1mNsfds7/RoJwor590BNYkLecF6yps%0A4+5FwFbgG0+QmdloM8sxs5zCwsL9CtO6SUNO69OBNplv7df+IiLpIMozhfOAoe7+k2D5EuA4dx+T%0A1GZR0CYvWP44aLNhb8fNzs72nJycSDKLiKSq2nCmsBbonLTcKVhXYZvg8lErYGOEmUREpBJRFoXZ%0AQJaZdTezhsBwYHq5NtOBy4LX5wGvV9afICIi0YrsiWZ3LzKzMcBMIAOY4O6Lzex2IMfdpwOPAo+b%0AWS6wiUThEBGRmEQ6zIW7zwBmlFt3a9Lr3cD5UWYQEZHw9PSWiIiUUVEQEZEyKgoiIlJGRUFERMpE%0A9vBaVMwsH/h0P3dvD+z1wbgUpc+cHvSZ08OBfOau7p5ZVaM6VxQOhJnlhHmiL5XoM6cHfeb0UBOf%0AWZePRESkjIqCiIiUSbeiMD7uADHQZ04P+szpIfLPnFZ9CiIiUrl0O1MQEZFKpE1RqGq+6FRjZp3N%0AbJaZLTGzxWY2Nu5MNcHMMsxsnpm9EHeWmmBmrc1smpktM7OlwYyHKc3MfhH8mV5kZk+aWcrNrWtm%0AE8xsfTDnTOm6tmb2ipl9FPxuE8V7p0VRCDlfdKopAq539z7AYODqNPjMAGOBpXGHqEF/Bl5y98OB%0Ao0nxz25mHYFrgWx370tiBOZUHF15IjC03LqbgdfcPQt4LViudmlRFIBBQK67r3T3PcAUYFjMmSLl%0A7p+7+9zg9TYS/1iUnw41pZhZJ+BM4JG4s9QEM2sFnERiCHrcfY+7b4k3VY2oDzQJJuZqCnwWc55q%0A5+5vkphOINkwYFLwehLwwyjeO12KQpj5olOWmXUD+gPvx5skcn8CfgmUxB2khnQH8oG/B5fMHjGz%0AZnGHipK7rwXuAVYDnwNb3f3leFPVmA7u/nnweh3QIYo3SZeikLbMrDnwNPBzd/8y7jxRMbOzgPXu%0APifuLDWoPnAs8KC79wd2ENElhdoiuI4+jERBPBRoZmYXx5uq5gUzVEZy62i6FIUw80WnHDNrQKIg%0APOHuz8SdJ2LfAs42s1UkLg9+z8z+EW+kyOUBee5eegY4jUSRSGVDgE/cPd/dC4FngBNizlRTvjCz%0AQwCC3+ujeJN0KQph5otOKWZmJK41L3X3P8adJ2rufou7d3L3biT+/77u7in9DdLd1wFrzOywYNUp%0AwJIYI9WE1cBgM2sa/Bk/hRTvXE+SPKf9ZcBzUbxJpNNx1hZ7my865lhR+xZwCfChmc0P1v0qmCJV%0AUsc1wBPBl52VwOUx54mUu79vZtOAuSTusJtHCj7ZbGZPAicD7c0sD/gN8HtgqpldQWKk6AsieW89%0A0SwiIqXS5fKRiIiEoKIgIiJlVBRERKSMioKIiJRRURARkTIqCiKAmRWb2fxg5M3nzax1DbznKjNr%0AH/X7iOwLFQWRhF3ufkww8uYm4Oq4A4nEQUVB5JveJRgw0RLuDs4gPjSzC4P1JyfP2WBmfzWzkcHr%0AVWZ2m5nNDfY5PFjfzsxeDgavexiwYH0zM3vRzBYE73NhDX9ekTIqCiJJgrk3TuGrYVDOAY4hMVfB%0AEODu0vFnqrDB3Y8FHgRuCNb9BngrGLxuOtAlWD8U+Mzdjw7OVF6qlg8jsh9UFEQSmgTDgWwE2gKv%0ABOtPBJ5092J3/wL4NzAwxPFKByCcA3QLXp8E/APA3V8ENgfrPwRONbO7zOzb7r71QD+MyP5SURBJ%0A2OXuxwBdgYZU3adQxNf//pSfErIg+F1MFWOMufsKEqObfgjcaWa3hg0tUt1UFESSBN/SrwWuD2b2%0A+g9wYTD3cyaJb/sfkBiQrI+ZNQruVDolxOHfBC4CMLPTgTbB60OBne7+DxITyKT68NdSi6XFKKki%0A+8Ld55nZQmAEics9xwMLSExq8stgyGrMbCqwEFhBYrTOqtwGPGlmc0lchlodrO9Hoq+iBCgErqrG%0AjyOyTzRKqoiIlNHlIxERKaOiICIiZVQURESkjIqCiIiUUVEQEZEyKgoiIlJGRUFERMqoKIiISJn/%0ADwTDYlOwveQfAAAAAElFTkSuQmCC%0A)

![](data:image/png;base64,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%0AEenx/wFnCA/wgYFyPAAAAABJRU5ErkJggg==%0A)

In [18]:

    select.get_info()

Out[18]:

rounds

mi\_mean

mi\_error

delta

num\_feat

features

0

0

0.000000

0.000000

0.000000

0

[]

1

1

0.157674

0.004439

0.000000

1

[3]

2

2

0.290224

0.005273

0.840660

2

[3, 0]

3

3

0.512264

0.005407

0.765067

3

[3, 0, 1]

4

4

0.618547

0.005094

0.207477

4

[3, 0, 1, 4]

5

5

0.757189

0.004030

0.224141

5

[3, 0, 1, 4, 2]

6

6

0.757529

0.004062

0.000449

6

[3, 0, 1, 4, 2, 6]

7

7

0.757945

0.004134

0.000549

7

[3, 0, 1, 4, 2, 6, 8]

8

8

0.758337

0.004186

0.000517

8

[3, 0, 1, 4, 2, 6, 8, 9]

9

9

0.758007

0.004290

-0.000435

9

[3, 0, 1, 4, 2, 6, 8, 9, 7]

10

10

0.757176

0.004434

-0.001096

10

[3, 0, 1, 4, 2, 6, 8, 9, 7, 5]

Selecionando

In [59]:

    X_new = select.transform(X, rd=5)

    X_new.shape

Out[59]:

    (15000, 5)

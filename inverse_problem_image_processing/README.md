<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Inverse problems in image processing](#inverse-problems-in-image-processing)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Inverse problems in image processing

- Grade
  - 50% 2 or 3 pratical sessions graded
  - 50% Exam


## Exam revision

Potential questions
- Define the $\ell_2$ norm
- Define the $\ell_1$ norm
- What is the idea of regularization?
- Define an algorithm to solve the problem proximal problem --> proximal gradient descent
- No convergence rates
- No proofs
- 1 part on regularization + 1 part on optimization
- $\nabla (\frac{1}{2} \|Ax - y\|^2)$
- Pourquoi régulariser ?
- Comment régulariser ?
- Itération de gradient stochatstique
- Calcul de proximal gradient (connaître la définition)
- Différence entre soft et hard thresholding
- Reprendre les définitions principales
- Que favorise la représentation en ondelettes ? 

### SGD, ISTA, FISTA, LISTA
- Si l'on a une régularisation l2, pas besoin de ISTA car on peut directement calculer le gradient de la fonction de coût. On peut donc directement utiliser SGD.
- Slide 9 et 17 de 04_prox sur les subdifferentials
- FB (synonyme de proximal gradient descent)):
  - 1 pas de gradient par rapport à la fonction lisse
  - Un pas de proximal gradient par rapport à la fonction non lisse
- ISTA:
  - Attention: $\tau$ dans le devoir 2 est un paramètre quelconque. Dans les diapositives, $\tau$ est un pas de gradient et il est multiplié par $\lambda$ qui est le paramètre de régularisation.
  - Cas particulier de FB avec $g = \ell_1$
- FISTA:
  - $g$ peut être n'importe quelle fonction convexe
  - On performe le prox sur le point issu de la version accélérée de la descente de gradient (Nesterov)
- LISTA:
  - Adaptation de FISTA avec apprentissage des hyper-paramètres à partir des données


## Questions

- What is a "Mother Wavelet"?
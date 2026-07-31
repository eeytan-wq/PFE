<h1 align="center">Schéma volumes finis well-balanced pour le modèle de Ripa</h1>

<h3 align="center">Écoulements à surface libre enrichis — vers les courants de turbidité</h3>

## Présentation

Ce dépôt regroupe le développement et l'implémentation d'un schéma de type **volumes finis** pour des modèles d'écoulements à surface libre enrichis, ainsi que sa validation sur une série de cas tests de complexité croissante, en vue d'applications aux **courants de turbidité**.

Le modèle classique de Saint-Venant, longtemps utilisé pour simuler les écoulements à surface libre, se limite à une vision homogène du fluide. Le **modèle de Ripa** en constitue une extension : il introduit une variable supplémentaire de température potentielle (ou de flottabilité) qui modifie la gravité effective, permettant de prendre en compte des hétérogénéités de température, de densité ou de topographie — cruciales dans de nombreux phénomènes géophysiques.

Ce modèle intermédiaire joue un double rôle dans l'étude :

- il est suffisamment simple pour être manipulé analytiquement, ce qui permet de dériver explicitement plusieurs familles d'états stationnaires au repos, de l'équilibre hydrostatique classique à des équilibres thermodynamiques plus complexes ;
- il contient déjà les difficultés essentielles des modèles de courants de turbidité : termes source liés à la topographie, équilibre entre pression hydrostatique et gravité, et nécessité de schémas rigoureusement **bien équilibrés** (*well-balanced*).

## Approche numérique

Cette richesse physique s'accompagne de défis numériques majeurs, notamment en matière de stabilité et de préservation des états d'équilibre. Le projet retrace l'évolution des stratégies numériques employées :

- mise en évidence des instabilités initiales et des limitations des méthodes classiques de reconstruction hydrostatique ;
- élaboration d'un schéma *well-balanced* d'ordre élevé basé sur une **formulation en perturbation**, qui traite à précision machine des écoulements aux conditions physiques complexes en isolant l'évolution des écarts à l'équilibre.

Le cadre traité est le **modèle de Ripa en une dimension**, sa discrétisation en volumes finis, puis la progression vers la préservation exacte de ses différents états stationnaires.

## Contenu du dépôt

Ce dépôt fournit le code des **cas tests fonctionnels** validés au cours du projet.

## Auteur

Eytanael Elleb Camille Pascal— Master 2 Ingénierie Mathématique, option Ingénierie Numérique.
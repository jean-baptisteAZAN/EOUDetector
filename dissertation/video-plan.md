# Video — 5 min max

Regle : montrer, pas lire. Chaque section finit sur un chiffre ou une image.

---

## 0:00 — Le probleme, en une scene (40 s)

- Lancer la demo. Epeler un nom au micro : "A - Z - A - N"
- Le systeme actuel coupe au milieu. Le laisser couper a l'ecran.
- "Un VAD n'entend que le silence. Une epellation, c'est du silence."
- Pourquoi ca compte ici : ligne medicale, on perd un nom, un numero, une date.

## 0:40 — Ce que j'ajoute (40 s)

- Schema du pipeline (figure 4.1 du memoire)
- Deux signaux au lieu d'un : le son (Smart Turn) + les mots (CamemBERT)
- Trois choses, dans cet ordre :
  1. fusionner les deux en une probabilite
  2. la **calibrer** pour qu'elle veuille dire quelque chose
  3. en tirer un seuil depuis un **cout**, pas un minuteur

## 1:20 — La demo qui marche (60 s)

- Rejouer la meme epellation. Le systeme attend.
- Montrer p_ac / p_lex / p_eou a l'ecran a chaque pause
- Un cas ou l'acoustique dit "fini" et le lexical dit "non" -> WAIT
- Un cas ou les deux disent "fini" -> END en 200 ms

## 2:20 — Les resultats (70 s)

- Tableau 5.1 a l'ecran
- **0.814 -> 0.897**, gain de 0.083, p < 1/3000, bootstrap par appel
- Honnete tout de suite : le lexical seul fait deja 0.845, donc le gain de la
  *fusion* est 0.052 sur la meilleure branche
- **La calibration** : tous les detecteurs externes du leaderboard sont mal
  calibres, aucun ne le rapporte. C'est la trouvaille, pas le 0.897.
- Un seuil pose sur une probabilite fausse ne peut pas etre relie a un cout.

## 3:30 — Ce qui n'a pas marche (60 s)

- Le morceau que personne ne met dans sa video. Le mettre.
- 709 appels medicaux reels, 8 kHz : l'acoustique tombe a **0.49**, le hasard
- Deux causes : bande etroite, et le plafond de la supervision faible
- Et le contre-exemple : en domaine, fusionner **degrade** (0.59 < 0.60)
- "Le pipeline sur etagere ne transfere pas. C'est le resultat le plus
  instructif du projet."

## 4:30 — La suite (30 s)

- Modele acoustique adapte au 8 kHz
- Un petit jeu annote a la main pour une mesure en domaine fiable
- Arret optimal sequentiel plutot qu'un seuil myope

---

## Ce qu'il faut absolument montrer

- [ ] la demo en direct, deux fois : le cas qui echoue, puis le cas qui marche
- [ ] les probabilites a l'ecran, pas juste la decision
- [ ] le tableau 5.1
- [ ] le 0.49 du gold

## Ce qu'il faut eviter

- lire le memoire a voix haute
- les 40 premieres secondes sur "le contexte du projet"
- finir sur les resultats positifs : finir sur le negatif est plus fort

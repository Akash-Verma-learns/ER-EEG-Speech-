"""
Genetic Algorithm feature selector.

fitness(chromosome) = Accuracy(SVM, selected_features) − λ · (n_selected / n_total)

Runs independently per modality (EEG and Speech) so the two modalities are
decoupled during selection — avoids cross-modal fitness leakage.
"""

import numpy as np
from typing import Optional, Tuple
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed


class GeneticAlgorithmSelector:
    def __init__(
        self,
        population_size: int = 50,
        n_generations: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.01,
        tournament_size: int = 3,
        sparsity_weight: float = 0.1,
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        self.pop_size = population_size
        self.n_gen = n_generations
        self.cx_rate = crossover_rate
        self.mut_rate = mutation_rate
        self.tour_size = tournament_size
        self.sparsity_weight = sparsity_weight
        self.n_jobs = n_jobs
        self.rng = np.random.RandomState(random_state)

        self.best_chromosome_: Optional[np.ndarray] = None
        self.best_fitness_: float = -np.inf
        self.selected_indices_: Optional[np.ndarray] = None
        self.fitness_history_: list = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GeneticAlgorithmSelector":
        """
        Parameters
        ----------
        X : (n_samples, n_features)
        y : (n_samples,)
        """
        n_features = X.shape[1]

        # Initialise population: each individual is a binary vector
        population = self._init_population(n_features)

        for gen in range(self.n_gen):
            fitnesses = self._evaluate_population(population, X, y)

            best_idx = np.argmax(fitnesses)
            if fitnesses[best_idx] > self.best_fitness_:
                self.best_fitness_ = fitnesses[best_idx]
                self.best_chromosome_ = population[best_idx].copy()

            self.fitness_history_.append(fitnesses.max())

            # Selection + crossover + mutation
            offspring = []
            while len(offspring) < self.pop_size:
                p1 = self._tournament_select(population, fitnesses)
                p2 = self._tournament_select(population, fitnesses)
                c1, c2 = self._crossover(p1, p2)
                offspring.extend([self._mutate(c1, n_features),
                                   self._mutate(c2, n_features)])

            # Elitism: keep best individual
            population = np.array(offspring[: self.pop_size])
            population[0] = self.best_chromosome_

            if (gen + 1) % 20 == 0:
                n_sel = self.best_chromosome_.sum()
                print(
                    f"  GA gen {gen+1:3d}/{self.n_gen}  "
                    f"best_fitness={self.best_fitness_:.4f}  "
                    f"n_selected={int(n_sel)}/{n_features}"
                )

        self.selected_indices_ = np.where(self.best_chromosome_)[0]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.selected_indices_ is None:
            raise RuntimeError("Call fit() first.")
        return X[:, self.selected_indices_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_population(self, n_features: int) -> np.ndarray:
        population = (self.rng.rand(self.pop_size, n_features) > 0.5).astype(np.int8)
        # Ensure each individual selects at least one feature
        for ind in population:
            if ind.sum() == 0:
                ind[self.rng.randint(n_features)] = 1
        return population

    def _evaluate_population(
        self, population: np.ndarray, X: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fitness)(chrom, X, y) for chrom in population
        )
        return np.array(results)

    def _fitness(
        self, chromosome: np.ndarray, X: np.ndarray, y: np.ndarray
    ) -> float:
        selected = np.where(chromosome)[0]
        if len(selected) == 0:
            return -1.0

        X_sel = X[:, selected]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sel)

        # 3-fold CV accuracy with a lightweight SVM (fast fitness proxy)
        skf = StratifiedKFold(n_splits=3, shuffle=False)
        accs = []
        for train_idx, val_idx in skf.split(X_scaled, y):
            clf = SVC(kernel="rbf", C=1.0, gamma="scale")
            clf.fit(X_scaled[train_idx], y[train_idx])
            acc = clf.score(X_scaled[val_idx], y[val_idx])
            accs.append(acc)

        accuracy = float(np.mean(accs))
        sparsity_penalty = self.sparsity_weight * (len(selected) / X.shape[1])
        return accuracy - sparsity_penalty

    def _tournament_select(
        self, population: np.ndarray, fitnesses: np.ndarray
    ) -> np.ndarray:
        candidates = self.rng.choice(len(population), self.tour_size, replace=False)
        winner = candidates[np.argmax(fitnesses[candidates])]
        return population[winner].copy()

    def _crossover(
        self, p1: np.ndarray, p2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.rng.rand() > self.cx_rate:
            return p1.copy(), p2.copy()
        n = len(p1)
        # Two-point crossover
        pts = sorted(self.rng.choice(n - 1, 2, replace=False) + 1)
        c1 = np.concatenate([p1[:pts[0]], p2[pts[0]:pts[1]], p1[pts[1]:]])
        c2 = np.concatenate([p2[:pts[0]], p1[pts[0]:pts[1]], p2[pts[1]:]])
        return c1, c2

    def _mutate(self, chromosome: np.ndarray, n_features: int) -> np.ndarray:
        mask = self.rng.rand(n_features) < self.mut_rate
        chromosome = chromosome.copy()
        chromosome[mask] ^= 1
        if chromosome.sum() == 0:
            chromosome[self.rng.randint(n_features)] = 1
        return chromosome

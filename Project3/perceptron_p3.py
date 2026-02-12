import numpy as np

class Perceptron:
    '''
    Version adaptée pour prendre en compte des poids d'échantillon
    et s'appliquer à un problème binaire (-1, +1).
    '''

    def __init__(self, alpha):
        if alpha <= 0:
            raise ValueError("alpha doit être strictement positif.")
        self.alpha = alpha
        self.w = None
        self.b = 0
        
    def train(self, X, y, epochs=1000, sample_weight=None):
        '''
        Entraînement d'un Perceptron binaire, 
        labels attendus : -1 ou +1.
        
        X : array (n_samples, n_features)
        y : array (n_samples,) avec valeurs dans {-1, +1}
        sample_weight : array (n_samples,) ou None
        '''
        n_samples, n_features = X.shape
        # initialisation
        self.w = np.zeros(n_features)
        self.b = 0.0

        if sample_weight is None:
            sample_weight = np.ones(n_samples)
        else:
            sample_weight = np.array(sample_weight)
            if len(sample_weight) != n_samples:
                raise ValueError("sample_weight doit avoir la même taille que X.")
        
        for _ in range(epochs):
            for i in range(n_samples):
                # calcul du score
                score = np.dot(X[i], self.w) + self.b
                predicted_label = 1 if score >= 0 else -1
                # mise à jour si erreur
                if predicted_label != y[i]:
                    self.w += self.alpha * sample_weight[i] * y[i] * X[i]
                    self.b += self.alpha * sample_weight[i] * y[i]
   
    def predict(self, X_new):
        '''
        Prévision de classes binaires (-1, +1).
        X_new : array (m_samples, n_features)
        '''
        scores = np.dot(X_new, self.w) + self.b
        y_hat = np.sign(scores)  # -1 ou +1
        return y_hat


class MultiClassPerceptronOvR:
    '''
    Implémente One-vs-Rest en encapsulant K Perceptrons,
    pour gérer un problème multi-classes avec labels 0..K-1.
    '''
    def __init__(self, alpha, n_classes, epochs=1000):
        self.alpha = alpha
        self.n_classes = n_classes
        self.epochs = epochs
        # Liste de perceptrons binaires, un pour chaque classe
        self.perceptrons = [Perceptron(alpha) for _ in range(n_classes)]
        
    def train(self, X, y, sample_weight=None):
        '''
        X : (n_samples, n_features)
        y : (n_samples,) avec étiquettes entières 0..(K-1).
        sample_weight : (n_samples,) ou None
        '''
        for k in range(self.n_classes):
            # construire les labels binaires pour la classe k
            y_k = np.where(y == k, 1, -1)
            # entraîner le perceptron k
            self.perceptrons[k].train(X, y_k, epochs=self.epochs, sample_weight=sample_weight)
    
    def predict(self, X_new):
        '''
        Retourne la classe prédite (0..K-1) par argmax du score.
        '''
        # calculer les scores de chaque perceptron
        scores = []
        for k in range(self.n_classes):
            # score_k : array shape (m_samples,)
            # on peut le calculer de la même manière que .predict, 
            # mais on récupère plutôt le "score" brut = dot + b
            score_k = np.dot(X_new, self.perceptrons[k].w) + self.perceptrons[k].b
            scores.append(score_k)
        # scores est une liste de length K, 
        # on fait un np.vstack pour obtenir (K, m_samples)
        all_scores = np.vstack(scores)  # shape (K, m_samples)
        
        # prédiction = argmax le long de la dimension K
        # axis=0 => on cherche pour chaque sample la classe max
        y_hat = np.argmax(all_scores, axis=0)
        return y_hat




class MultiClassAdaBoostSAMME:
    def __init__(self, base_estimator_class,  # p.ex MultiClassPerceptronOvR
                 n_classes,                   # nombre de classes
                 M=10,                        # nb d'itérations de boosting
                 alpha=1.0,                   # param. perceptron
                 epochs=1000):
        self.base_estimator_class = base_estimator_class
        self.n_classes = n_classes
        self.M = M
        self.alpha_perceptron = alpha
        self.epochs = epochs
        
        self.estimators = []   # liste pour stocker T^(m)
        self.alphas = []       # liste pour stocker alpha^(m)
    
    def fit(self, X, y):
        n_samples = X.shape[0]
        # Initialiser w_i = 1/n
        w = np.ones(n_samples) / n_samples
        
        for m in range(self.M):
            # 1) Entraîner un nouveau classifieur faible sur (X, y, w)
            estimator = self.base_estimator_class(self.alpha_perceptron,
                                                  self.n_classes,
                                                  epochs=self.epochs)
            estimator.train(X, y, sample_weight=w)
            
            # 2) Calculer l'erreur (err_m)
            #   err_m = somme des w_i pour les i mal classés
            predictions = estimator.predict(X)  # classes prédites
            incorrect = (predictions != y)
            err_m = np.sum(w * incorrect) / np.sum(w)
            
            # 3) Calculer alpha_m
            #   SAMME : alpha_m = ln((1-err_m)/err_m) + ln(K-1)
            #   (assure que le seuil pour être "meilleur que hasard" est 1/K)
            eps = 1e-12  # pour éviter log(0)
            alpha_m = np.log((1 - err_m + eps) / (err_m + eps)) + np.log(self.n_classes - 1)
            
            # 4) Mettre à jour w_i
            #   w_i <- w_i * exp(alpha_m * 1(incorrect_i))
            #   puis normaliser
            w = w * np.exp(alpha_m * incorrect)
            
            w_sum = np.sum(w)
            if w_sum == 0:
                # si plus aucun poids, on arrête
                break
            w /= w_sum
            
            # stocker
            self.estimators.append(estimator)
            self.alphas.append(alpha_m)
    
    def predict(self, X):
        # On combine selon :
        #   C(x) = argmax_k sum_{m} [ alpha_m * 1(T^m(x) = k) ]
        # => on calcule "score_k(x) = sum_{m, T^m(x) = k} alpha_m"
        
        # Pour accélérer, on va boucler sur M estimators
        # puis "voter" pour la classe prédite par chacun
        n_samples = X.shape[0]
        class_scores = np.zeros((n_samples, self.n_classes))
        
        for m, estimator in enumerate(self.estimators):
            pred_m = estimator.predict(X)  # shape (n_samples,)
            alpha_m = self.alphas[m]
            # Ajouter alpha_m au score de la classe prédite
            for i in range(n_samples):
                class_scores[i, pred_m[i]] += alpha_m
        
        # la classe finale est celle avec le plus grand score
        return np.argmax(class_scores, axis=1)

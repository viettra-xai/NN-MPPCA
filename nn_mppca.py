import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from sklearn.metrics import precision_recall_fscore_support




def random_batch(X, y=None, batch_size=32):
    idx = np.random.randint(len(X), size=batch_size)
    return X[idx]

def progress_bar(iteration, total, size=30):
    running = iteration < total
    c = ">" if running else "="
    p = (size - 1) * iteration // total
    fmt = "{{:-{}d}}/{{}} [{{}}]".format(len(str(total)))
    params = [iteration, total, "=" * p + c + "." * (size - p - 1)]
    return fmt.format(*params)

def print_status_bar(iteration, total, loss, metrics=None, size=30):
    metrics = " - ".join(["{}: {:.4f}".format(m.name, m.result()) for m in [loss] + (metrics or [])])
    end = "" if iteration < total else "\n"
    print("\r{} - {}".format(progress_bar(iteration, total), metrics), end=end)
    
def euclid_norm(x):
    return tf.sqrt(tf.reduce_sum(tf.square(x), axis=1))


class Estimator(keras.layers.Layer):
    # estimation network
    def __init__(self, hidden_layer_sizes, activation="elu", kernel_initializer="he_normal", dropout_rate=None, **kwargs):
        super().__init__(**kwargs)
        self.dropout_layer = keras.layers.Dropout(rate=dropout_rate)
        self.hidden = [keras.layers.Dense(size, activation=activation, kernel_initializer=kernel_initializer,
                                          kernel_regularizer=keras.regularizers.l2(0.01))
                               for size in hidden_layer_sizes[:-1]]
        self.out = keras.layers.Dense(hidden_layer_sizes[-1], activation=keras.activations.softmax,
                                      kernel_initializer=kernel_initializer, kernel_regularizer=keras.regularizers.l2(0.01))

    def call(self, z):
        for layer in self.hidden:
            z = layer(z)
            z = self.dropout_layer(z)
        output = self.out(z)
        return output


class NNMPPCA:
    """Neural network -based Mixture of Probabilistic Principal Component Analyzers.
    """

    MODEL_FILENAME = "MPPCA_model"
    SCALER_FILENAME = "MPPCA_scaler"

    def __init__(self, input_size, est_hiddens, est_activation, kernel_initializer, est_dropout_ratio=0.2,
                 n_epochs=1000, batch_size = 128, lambda1=1, lambda2=0.1, lambda3=0.01, learning_rate=0.001, 
                 patience=10, normalize=True, random_seed=42):
        

        inputs = keras.layers.Input(input_size, name="input")
        self.est_network = Estimator(est_hiddens, est_activation, kernel_initializer, est_dropout_ratio)
        gamma = self.est_network(inputs)
        self.mppca= keras.models.Model(inputs=[inputs], outputs=[gamma])

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        self.W = None
        self.sigma2 = None
        
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        
        self.lr = learning_rate
        self.patience = patience
        
        self.normalize = normalize
        self.scaler = None
        self.seed = random_seed
        
        self.phi_list = []
        self.mu_list = []
        self.L_list = []


    def custom_loss(self, inputs):
        # calculate the loss function with three loss components
        
        # (1) reconstruction probability loss
        res_probs, A, C, latent_error = self.res_prob(inputs)
        res_prob_loss = self.lambda1 * tf.reduce_mean(res_probs)
 
        # (2) reconstruction loss of MMPCA 
        pca_loss = self.lambda2 * tf.reduce_mean(latent_error, axis=0)
        
        # (3) sigularity penalty
#         diag_loss = tf.reduce_sum(tf.divide(1, tf.linalg.diag_part(A))) + tf.reduce_sum(tf.divide(1, tf.linalg.diag_part(C)))
        diag_loss = tf.reduce_sum(tf.divide(1, tf.linalg.diag_part(A)))
        cov_loss = self.lambda3 * diag_loss
        
        # sum of three loss components
        loss = res_prob_loss + pca_loss + cov_loss

        return loss
    
    def para_init(self, inputs): 
        
        if self.normalize:
            self.scaler = StandardScaler().fit(inputs)
            inputs = self.scaler.transform(inputs)

        self.d = d = inputs.shape[-1]
        gamma = self.est_network(inputs)
        
        # Calculate mu, sigma
        # i   : index of samples
        # k   : index of components
        # l,m : index of features
        gamma_sum = tf.reduce_sum(gamma, axis=0)
        phi = tf.reduce_mean(gamma, axis=0)
        mu =  tf.einsum('ik,il->kl', gamma, inputs) / gamma_sum[:, np.newaxis]
        z_centered_1 = tf.sqrt(gamma[:, :, np.newaxis]) * (inputs[:, np.newaxis, :] - mu[np.newaxis, :, :])
        S =  tf.einsum('ikl,ikm->klm', z_centered_1, z_centered_1) / gamma_sum[:, np.newaxis, np.newaxis] 
        if round(d/2) > 10:
            q = 3
        else:
            q = round(d/2)
            
        self.q = q 
        
        # Parameter Initialization
        
        eig, vec = tf.linalg.eigh(S) # eigenvalues are sorted in non-descending order
        eig, vec = eig[:,::-1], vec[:,:,::-1] # reorder eigencalues and eigenvectors in descending direction
        self.sigma2 = tf.square(tf.reduce_mean(eig[:, q:], axis=1)) # noise variance
        U = vec[:, :, :q] # matrix of q eigvectors of covariance matrix
        K = tf.linalg.diag(eig[:, :q]) # diagonal matrix of q largest eigenvalues
        K_centered = tf.sqrt(K - self.sigma2[:, np.newaxis, np.newaxis]*tf.linalg.eye(q))
        self.W = tf.einsum('ikl,ilm->ikm', U, K_centered) 
        
        
    def res_prob(self, inputs):
        """ calculate an energy of each row of z

        Parameters
        ----------
        z : tf.Tensor, shape (n_samples, n_features)
            data each row of which is calculated its energy.

        Returns
        -------
        reconstruction probability or anomaly score : tf.Tensor, shape (n_samples)

        """
        q = self.q
        d = self.d
        gamma = self.est_network(inputs)
        
        # Calculate mu, sigma
        # i   : index of samples
        # k   : index of components
        # l,m : index of features
        gamma_sum = tf.reduce_sum(gamma, axis=0)
        self.phi = phi = tf.reduce_mean(gamma, axis=0)
        self.mu = mu = tf.einsum('ik,il->kl', gamma, inputs) / gamma_sum[:, np.newaxis]
        z_centered_1 = tf.sqrt(gamma[:, :, np.newaxis]) * (inputs[:, np.newaxis, :] - mu[np.newaxis, :, :])
        S =  tf.einsum('ikl,ikm->klm', z_centered_1, z_centered_1) / gamma_sum[:, np.newaxis, np.newaxis] 
        SW = tf.einsum('ikl,ilm->ikm', S, self.W) 
        
        sigma2I = self.sigma2[:, np.newaxis, np.newaxis]*tf.linalg.eye(q)
        self.M = sigma2I + tf.einsum('ikl,ilm->ikm', tf.transpose(self.W, [0, 2, 1]), self.W)
        self.Minv = tf.linalg.inv(self.M)
        self.MinvWT = tf.einsum('ikl,ilm->ikm', self.Minv, tf.transpose(self.W, [0, 2, 1]))
        min_vals_W = tf.linalg.diag(tf.ones(q, dtype=tf.float32)) * 1e-3
        A = self.sigma2[:, np.newaxis, np.newaxis]*tf.linalg.eye(q) + tf.einsum('ikl,ilm->ikm', self.MinvWT, SW)
        Ainv = tf.linalg.inv(A + min_vals_W[np.newaxis, :, :])
        
        W_new = tf.einsum('ikl,ilm->ikm', SW, Ainv)
        MinvWnT = tf.einsum('ikl,ilm->ikm', self.Minv, tf.transpose(W_new, [0, 2, 1]))
        sigma2_new = 1/d*tf.linalg.trace(S - tf.einsum('ikl,ilm->ikm', SW, MinvWnT))
        self.W = W_new
        self.sigma2 = sigma2_new
        

        # model covariance
        C = (self.sigma2[:, np.newaxis, np.newaxis]*tf.linalg.eye(d) + 
             tf.einsum('ikl,ilm->ikm', self.W, tf.transpose(self.W, [0, 2, 1])))
  

        # Calculate a cholesky decomposition of model covariance

        min_vals = tf.linalg.diag(tf.ones(d, dtype=tf.float32)) * 1e-4
        self.L = L = tf.linalg.cholesky(C + min_vals[np.newaxis, :, :])
        z_centered_2 = inputs[:, np.newaxis, :] - mu[np.newaxis, :, :] 
        v = tf.linalg.triangular_solve(L, tf.transpose(z_centered_2, [1, 2, 0]))  # kli

        # log(det(cov)) = 2 * sum[log(diag(L))]
        log_det_cov = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)), axis=1)

        # To calculate energies, use "log-sum-exp"
        logits = tf.math.log(phi[:, np.newaxis]) - 0.5 * (tf.reduce_sum(tf.square(v), axis=1)
                                                          + d * tf.math.log(
                    2.0 * tf.constant(np.pi, dtype="float32")) + log_det_cov[:, np.newaxis])
        res_probs = - tf.reduce_logsumexp(logits, axis=0)
        
        # Calculate the reconstruction error of MMPCA
        
        W2 = tf.einsum('ikl,ilm->ikm', tf.transpose(self.W, [0, 2, 1]), self.W) # calculate W'*W
        min_vals_W2 = tf.linalg.diag(tf.ones(q, dtype=tf.float32)) * 1e-5
        W2inv = tf.linalg.inv(W2 + min_vals_W2)
        WW2inv = tf.einsum('ikl,ilm->ikm', self.W, W2inv)

        B = tf.einsum('ikl, ilm->ikm', WW2inv, tf.transpose(self.W, [0, 2, 1])) # transition matrix
        z_i = tf.einsum('ikl, ilm->ikm', B, tf.transpose(z_centered_2, [1, 2, 0])) + mu[:, :, np.newaxis]# individual component reconstruction of z 
        z_res = tf.einsum('ik, ikm->im', gamma, tf.transpose(z_i, [2, 0, 1])) # reconstruction of z
        latent_error = tf.reduce_sum(tf.square(inputs - z_res), axis=1) # reconstruction error of latent variables

        return res_probs, A, C, latent_error


    def fit(self, inputs, X_test, y_test):
        tf.random.set_seed(self.seed)
        np.random.seed(seed=self.seed)
        if self.normalize:
            self.scaler = MinMaxScaler().fit(inputs)
            inputs = self.scaler.transform(inputs)
        X_train, X_valid = train_test_split(inputs, test_size=0.3, random_state=42)

        n_steps = len(X_train) // self.batch_size
        optimizer = keras.optimizers.Nadam(learning_rate=self.lr)
#         optimizer = keras.optimizers.Adamax(learning_rate=self.lr)
        # loss_fn = keras.losses.mean_squared_error
        mean_loss = keras.metrics.Mean(name='mean_loss')
        metrics = keras.metrics.Mean(name='val_loss')
        minimum_val_loss = float("inf")
        best_epoch = 1
        best_model = None
        wait = 0

        for epoch in range(1, self.n_epochs + 1):
            print("Epoch {}/{}".format(epoch, self.n_epochs))
            for step in range(1, n_steps + 1):
                X_batch = random_batch(X_train, batch_size=self.batch_size)

                with tf.GradientTape() as tape:
                    main_loss = self.custom_loss(X_batch)
                    loss = tf.add_n([main_loss] + self.mppca.losses)
                gradients = tape.gradient(loss, self.mppca.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.mppca.trainable_variables))
                
                for variable in self.mppca.variables:
                    if variable.constraint is not None:
                        variable.assign(variable.constraint(variable))
                mean_loss(loss)
                print_status_bar(step * self.batch_size, len(inputs), mean_loss)
            val_loss = tf.add_n([self.custom_loss(X_valid)] + self.mppca.losses)
            wait +=1
            print('\n wait:', wait)
#             if val_loss < minimum_val_loss and val_loss >= minimum_val_loss - 1:
#                 minimum_val_loss = val_loss
            if val_loss < minimum_val_loss - 0.2:
                minimum_val_loss = val_loss
                self.best_epoch = best_epoch = epoch
                self.mppca.save_weights("my_keras_weights.ckpt")
                wait = 0
            if wait >= self.patience:
                break 
            mean_loss(self.custom_loss(X_batch))
            metrics(val_loss)
            print_status_bar(len(inputs), len(inputs), mean_loss, [metrics])
            for metric in [mean_loss] + [metrics]:
                metric.reset_states()
            
#             # In case we want to measure f1score of the model after each training epoch.
#             # The fit function is now fit(self, inputs, X_test, y_test)
            self.metrics_cal(X_test, y_test) 
            self.mu_list.append(self.mu)
            self.phi_list.append(self.phi)
            self.L_list.append(self.L)

#         print('load weights of best epoch:', self.best_epoch)
            if val_loss < minimum_val_loss - 2:
                break
        self.mppca.load_weights("my_keras_weights.ckpt")

    
    def validate(self, inputs):
        # calculate the energy of input samples
        if self.normalize:
            inputs = self.scaler.transform(inputs)

        z_centered = inputs[:, np.newaxis, :] - self.mu[np.newaxis, :, :] 
        v = tf.linalg.triangular_solve(self.L, tf.transpose(z_centered, [1, 2, 0]))  # kli

        # log(det(cov)) = 2 * sum[log(diag(L))]
        log_det_cov = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(self.L)), axis=1)

        # To calculate energies, use "log-sum-exp"
        logits = tf.math.log(self.phi[:, np.newaxis]) - 0.5 * (tf.reduce_sum(tf.square(v), axis=1)
                                                          + self.d * tf.math.log(
                    2.0 * tf.constant(np.pi, dtype="float32")) + log_det_cov[:, np.newaxis])
        res_probs = - tf.reduce_logsumexp(logits, axis=0)

        return res_probs.numpy()
    
    def predict(self, inputs):
        # calculate the energy of input samples
        phi = self.phi_list[self.best_epoch-1]
        mu = self.mu_list[self.best_epoch-1]
        L = self.L_list[self.best_epoch-1]
        if self.normalize:
            inputs = self.scaler.transform(inputs)
        
        z_centered = inputs[:, np.newaxis, :] - mu[np.newaxis, :, :] 
        v = tf.linalg.triangular_solve(L, tf.transpose(z_centered, [1, 2, 0]))  # kli

        # log(det(cov)) = 2 * sum[log(diag(L))]
        log_det_cov = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)), axis=1)

        # To calculate energies, use "log-sum-exp"
        logits = tf.math.log(phi[:, np.newaxis]) - 0.5 * (tf.reduce_sum(tf.square(v), axis=1)
                                                          + self.d * tf.math.log(
                    2.0 * tf.constant(np.pi, dtype="float32")) + log_det_cov[:, np.newaxis])
        res_probs = - tf.reduce_logsumexp(logits, axis=0)

        return res_probs.numpy()
    
    def metrics_cal(self, X_test, y_test):
        y_pred = self.validate(X_test)
        # Energy thleshold to detect anomaly = 80% percentile of energies
        anomaly_energy_threshold = np.percentile(y_pred, 80)
        print(f"Energy thleshold to detect anomaly : {anomaly_energy_threshold:.3f}")
        # Detect anomalies from test data
        y_pred_flag = np.where(y_pred >= anomaly_energy_threshold, 1, 0)
        prec, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred_flag, average="binary")
        print(f" Precision = {prec:.3f}")
        print(f" Recall    = {recall:.3f}")
        print(f" F1-Score  = {fscore:.3f}")

    def restore(self):
        model = self.damppca
        return model
    
    
    


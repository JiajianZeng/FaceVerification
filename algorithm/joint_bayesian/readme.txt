Algorithm Joint Bayesian is implemented according to paper 'Bayesian Face Revisited: A Joint Formulation'. Plz refer to the paper for details. Following are some concise informations about my implementation.

(1)joint_bayesian.m ------ train joint bayesian model, and return the parameters.
(2)joint_bayesian_distance.m ------ compute distance between two feature vectors using the parameters obtained from (1).
(3)svm.m ------ train svm model using distances obtained from (2) as train data and labels, and return the parameters.
(4)svm_verification_demo.m ------ a demo application indicating how to train Joint Bayesian model and SVM model, and classify unseen samples using SVM model obtained from (3).

Before run svm_verification_demo application, you must install libsvm under your matlab first. 
